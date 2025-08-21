import logging
import os
import time

from pathvalidate import sanitize_filepath
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.annotation_readers.geti_annotation_reader import GetiAnnotationReader
from geti_sdk.data_models import Dataset
from geti_sdk.data_models.containers.media_list import MediaList
from geti_sdk.data_models.enums.dataset_format import DatasetFormat
from geti_sdk.data_models.enums.include_models import IncludeModelsType
from geti_sdk.data_models.media import Image, Video
from geti_sdk.data_models.project import Project
from geti_sdk.http_session.geti_session import GetiSession
from geti_sdk.import_export.tus_uploader import TUSUploader
from geti_sdk.platform_versions import GETI_25_VERSION
from geti_sdk.rest_clients import ConfigurationClient, ProjectConfigurationClient
from geti_sdk.rest_clients.annotation_clients.annotation_client import AnnotationClient
from geti_sdk.rest_clients.dataset_client import DatasetClient
from geti_sdk.rest_clients.media_client.image_client import ImageClient
from geti_sdk.rest_clients.media_client.video_client import VideoClient
from geti_sdk.rest_clients.model_client import ModelClient
from geti_sdk.rest_clients.prediction_client import PredictionClient
from geti_sdk.rest_clients.project_client.project_client import ProjectClient
from geti_sdk.utils.job_helpers import get_job_with_timeout, monitor_job
from geti_sdk.utils.project_helpers import get_project_folder_name


class GetiIE:
    """
    Class to handle importing and exporting projects and datasets to and from the Intel® Geti™ platform.
    """

    def __init__(self, workspace_id: str, session: GetiSession, project_client: ProjectClient) -> None:
        """
        Initialize the GetiIE class.

        :param workspace_id: The workspace id.
        :param session: The Geti session.
        :param project_client: The project client.
        """
        self.workspace_id = workspace_id
        self.session = session
        self.base_url = f"workspaces/{workspace_id}/"
        self.project_client = project_client

    def download_project_data(
        self,
        project: Project,
        target_folder: str | None = None,
        include_predictions: bool = False,
        include_active_models: bool = False,
        max_threads: int = 10,
    ) -> Project:
        """
        Download a project from the Geti Platform.

        :param project_name: The name of the project.
        :param project_id: The id of the project.
        :param target_folder: The path to save the downloaded project.
        :param include_predictions: Whether to download predictions for the project.
        :param include_active_models: Whether to download active models for the project.
        :param max_threads: The maximum number of threads to use for downloading media.
        :return: The downloaded project.
        """
        # Validate or create target_folder
        if target_folder is None:
            target_folder = os.path.join(".", get_project_folder_name(project))
        else:
            sanitize_filepath(target_folder, platform="auto")
        os.makedirs(target_folder, exist_ok=True, mode=0o770)

        # Download project creation parameters:
        self.project_client.download_project_info(project=project, path_to_folder=target_folder)

        # Download images
        image_client = ImageClient(workspace_id=self.workspace_id, session=self.session, project=project)
        images = image_client.get_all_images()
        if len(images) > 0:
            image_client.download_all(
                path_to_folder=target_folder,
                append_image_uid=images.has_duplicate_filenames,
                max_threads=max_threads,
            )

        # Download videos
        video_client = VideoClient(workspace_id=self.workspace_id, session=self.session, project=project)
        videos = video_client.get_all_videos()
        if len(videos) > 0:
            video_client.download_all(
                path_to_folder=target_folder,
                append_video_uid=videos.has_duplicate_filenames,
                max_threads=max_threads,
            )

        # Download annotations
        annotation_client = AnnotationClient(session=self.session, project=project, workspace_id=self.workspace_id)
        annotation_client.download_all_annotations(path_to_folder=target_folder, max_threads=max_threads)

        # Download predictions
        prediction_client = PredictionClient(workspace_id=self.workspace_id, session=self.session, project=project)
        if prediction_client.ready_to_predict and include_predictions:
            if len(images) > 0:
                prediction_client.download_predictions_for_images(
                    images=images,
                    path_to_folder=target_folder,
                    include_result_media=True,
                )
            if len(videos) > 0:
                prediction_client.download_predictions_for_videos(
                    videos=videos,
                    path_to_folder=target_folder,
                    include_result_media=True,
                    inferred_frames_only=False,
                )

        # Download active models
        if include_active_models:
            model_client = ModelClient(workspace_id=self.workspace_id, session=self.session, project=project)
            model_client.download_all_active_models(path_to_folder=target_folder)

        return project

    def upload_project_data(
        self,
        target_folder: str,
        project_name: str | None = None,
        enable_auto_train: bool = True,
        max_threads: int = 5,
    ) -> Project:
        """
        Upload a project to the Geti Platform.

        :param target_folder: The path to the project data folder.
        :param project_name: The name of the project.
        :param enable_auto_train: Whether to enable auto-train for the project.
        :param max_threads: The maximum number of threads to use for uploading media.
        :return: The uploaded project.
        """
        project = self.project_client.create_project_from_folder(
            path_to_folder=target_folder, project_name=project_name
        )

        # Disable auto-train to prevent the project from training right away
        if self.session.version.is_configuration_revamped:
            # Geti version >= 2.12
            configuration_client = ProjectConfigurationClient(
                workspace_id=self.workspace_id, session=self.session, project=project
            )
            configuration_client.set_project_auto_train(auto_train=False)
        else:
            configuration_client = ConfigurationClient(
                workspace_id=self.workspace_id, session=self.session, project=project
            )
            configuration_client.set_project_auto_train(auto_train=False)

        # Upload media
        image_client = ImageClient(workspace_id=self.workspace_id, session=self.session, project=project)
        video_client = VideoClient(workspace_id=self.workspace_id, session=self.session, project=project)

        # Check the media folders inside the project folder. If they are organized
        # according to the projects datasets, upload the media into their corresponding
        # dataset. Otherwise, upload all media into training dataset.
        dataset_client = DatasetClient(workspace_id=self.workspace_id, session=self.session, project=project)
        if not dataset_client.has_dataset_subfolders(target_folder):
            # Upload all media directly to the training dataset
            images = image_client.upload_folder(
                path_to_folder=os.path.join(target_folder, "images"),
                max_threads=max_threads,
            )
            videos = video_client.upload_folder(
                path_to_folder=os.path.join(target_folder, "videos"),
                max_threads=max_threads,
            )
        else:
            # Make sure that media is uploaded to the correct dataset
            images: MediaList[Image] = MediaList([])
            videos: MediaList[Video] = MediaList([])
            for dataset in project.datasets:
                images.extend(
                    image_client.upload_folder(
                        path_to_folder=os.path.join(target_folder, "images", dataset.name),
                        dataset=dataset,
                        max_threads=max_threads,
                    )
                )
                videos.extend(
                    video_client.upload_folder(
                        path_to_folder=os.path.join(target_folder, "videos", dataset.name),
                        dataset=dataset,
                        max_threads=max_threads,
                    )
                )

        # Short sleep to make sure all uploaded media is processed server side
        time.sleep(5)

        # Upload annotations
        annotation_reader = GetiAnnotationReader(
            base_data_folder=os.path.join(target_folder, "annotations"),
            task_type=None,
            anomaly_reduction=(self.session.version >= GETI_25_VERSION),
        )
        annotation_client = AnnotationClient[GetiAnnotationReader](
            session=self.session,
            project=project,
            workspace_id=self.workspace_id,
            annotation_reader=annotation_reader,
        )
        if len(images) > 0:
            annotation_client.upload_annotations_for_images(
                images=images,
                max_threads=max_threads,
            )
        if len(videos) > 0:
            are_videos_processed = False
            start_time = time.time()
            logging.info("Waiting for the Geti server to process all uploaded videos...")
            while (not are_videos_processed) and (time.time() - start_time < 100):
                # Ensure all uploaded videos are processed by the server
                project_videos = video_client.get_all_videos()
                uploaded_ids = {video.id for video in videos}
                project_video_ids = {video.id for video in project_videos}
                are_videos_processed = uploaded_ids.issubset(project_video_ids)
                time.sleep(1)
            annotation_client.upload_annotations_for_videos(videos=videos, max_threads=max_threads)

        configuration_client.set_project_auto_train(auto_train=enable_auto_train)
        logging.info(f"Project '{project.name}' was uploaded successfully.")
        return project

    def download_all_projects(
        self, target_folder: str = "./projects", include_predictions: bool = True
    ) -> list[Project]:
        """
        Download all projects from the Geti Platform.

        :param target_folder: The path to the directory to save the downloaded projects.
        :param include_predictions: Whether to download predictions for the projects.
        :return: The downloaded projects.
        """
        # Obtain project details from cluster
        projects = self.project_client.get_all_projects()

        # Validate or create target_folder
        os.makedirs(target_folder, exist_ok=True, mode=0o770)
        logging.info(
            f"Found {len(projects)} projects in the designated workspace on the "
            f"Intel® Geti™ server. Commencing project download..."
        )

        # Download all found projects
        with logging_redirect_tqdm(tqdm_class=tqdm):
            for index, project in enumerate(tqdm(projects, desc="Downloading projects")):
                logging.info(f"Downloading project '{project.name}'... {index + 1}/{len(projects)}.")
                self.download_project_data(
                    project=project,
                    target_folder=os.path.join(target_folder, get_project_folder_name(project)),
                    include_predictions=include_predictions,
                )
        return projects

    def upload_all_projects(self, target_folder: str) -> list[Project]:
        """
        Upload all projects in the target directory to the Geti Platform.

        :param target_folder: The path to the directory containing the project data folders.
        :return: The uploaded projects.
        """
        candidate_project_folders = [os.path.join(target_folder, subfolder) for subfolder in os.listdir(target_folder)]
        project_folders = [folder for folder in candidate_project_folders if ProjectClient._is_project_dir(folder)]
        logging.info(
            f"Found {len(project_folders)} project data folders in the target "
            f"directory '{target_folder}'. Commencing project upload..."
        )
        projects: list[Project] = []
        with logging_redirect_tqdm(tqdm_class=tqdm):
            for index, project_folder in enumerate(tqdm(project_folders, desc="Uploading projects")):
                logging.info(
                    f"Uploading project from folder '{os.path.basename(project_folder)}'..."
                    f" {index + 1}/{len(project_folders)}."
                )
                project = self.upload_project_data(target_folder=project_folder, enable_auto_train=False)
                projects.append(project)
        return projects

    def import_dataset_as_new_project(self, filepath: os.PathLike, project_name: str, project_type: str) -> Project:
        """
        Import a dataset as a new project to the Geti Platform.

        :param filepath: The path to the dataset archive.
        :param project_name: The name of the new project.
        :param project_type: The type of the new project. Provide one of
            [classification, classification_hierarchical, detection, segmentation, instance_segmentation,
            anomaly_classification, anomaly_detection, anomaly_segmentation, anomaly,
            detection_oriented, detection_to_classification, detection_to_segmentation]
        :return: The imported project.
        :raises: RuntimeError if the project type is not supported for the imported dataset.
        """
        # Upload the dataset archive to the server
        upload_endpoint = self.base_url + "datasets/uploads/resumable"
        file_id = self._tus_upload_file(upload_endpoint=upload_endpoint, filepath=filepath)
        # Prepare for import
        response = self.session.get_rest_response(
            url=f"{self.base_url}datasets:prepare-for-import?file_id={file_id}",
            method="POST",
        )
        job = get_job_with_timeout(
            job_id=response["job_id"],
            session=self.session,
            workspace_id=self.workspace_id,
            job_type="import_dataset",
        )
        job = monitor_job(session=self.session, job=job, interval=5)
        # Make sure that the project type is supported for the imported dataset
        if "_to_" in project_type:
            # Translate the SDK `detection_to_segmentation` project type format
            # to the Geti Platform `detection_segmentation` format
            project_type = project_type.replace("_to_", "_")
        project_dict = next(
            (entry for entry in job.metadata.supported_project_types if entry["project_type"] == project_type),
            None,
        )
        if project_dict is None:
            supported_project_types = [entry["project_type"] for entry in job.metadata.supported_project_types]
            raise RuntimeError(
                f"Project type '{project_type}' is not supported for the imported dataset.\n"
                f" Please select one of the supported project types: `{supported_project_types}`"
            )
        # Create a new project from the imported dataset
        label_names = [
            label_dict["name"] for task_dict in project_dict["pipeline"]["tasks"] for label_dict in task_dict["labels"]
        ]
        data = {
            "project_name": project_name,
            "task_type": project_type,
            "file_id": job.metadata.file_id,
            "labels": [{"name": label_name} for label_name in label_names],
        }
        response = self.session.get_rest_response(
            url=f"{self.base_url}projects:import-from-dataset",
            method="POST",
            data=data,
        )
        # Get the job id and monitor the job
        # until it returns the project id
        job = get_job_with_timeout(
            job_id=response["job_id"],
            session=self.session,
            workspace_id=self.workspace_id,
            job_type="import_project_from_dataset",
        )
        job = monitor_job(session=self.session, job=job, interval=5)
        logging.info(f"Project '{project_name}' was successfully imported from the dataset.")
        imported_project = self.project_client.get_project(
            project_id=job.metadata.project_id,
        )
        if imported_project is None:
            raise RuntimeError(f"Failed to retrieve the imported project '{project_name}'.")
        return imported_project

    def import_project(self, filepath: os.PathLike, project_name: str | None = None) -> Project:
        """
        Import a project to the Geti Platform.

        :param filepath: The path to the project archive.
        :param project_name: The name of the project.
        :return: The imported project.
        """
        if project_name is None:
            project_name = os.path.basename(filepath).split(".")[0]

        upload_endpoint = self.base_url + "projects/uploads/resumable"
        file_id = self._tus_upload_file(upload_endpoint=upload_endpoint, filepath=filepath)

        # Start project import process using the uploaded archive
        response = self.session.get_rest_response(
            url=f"{self.base_url}projects:import",
            method="POST",
            data={
                "file_id": file_id,
                "project_name": project_name,
            },
        )

        job = get_job_with_timeout(
            job_id=response["job_id"],
            session=self.session,
            workspace_id=self.workspace_id,
            job_type="import_project",
        )

        job = monitor_job(session=self.session, job=job, interval=5)
        if not job.metadata.project.id:
            # Wait a few seconds for the metadata to become in sync
            time.sleep(3)
            job = get_job_with_timeout(
                job_id=job.id,
                session=self.session,
                workspace_id=self.workspace_id,
                job_type="import_project",
            )
        imported_project = self.project_client.get_project(
            project_id=job.metadata.project.id,
        )
        if imported_project is None:
            raise RuntimeError(f"Failed to retrieve the imported project '{project_name}'.")
        return imported_project

    def _tus_upload_file(self, upload_endpoint: str, filepath: os.PathLike) -> str:
        """
        Upload a file using the TUS protocol.

        :param upload_endpoint: The TUS upload endpoint.
        :param filepath: The path to the file to upload.
        :return: The file id created on the Geti Platform.
        :raises: RuntimeError if the file id is not retrieved.
        """
        tus_uploader = TUSUploader(session=self.session, base_url=upload_endpoint, file_path=filepath)
        tus_uploader.upload()
        file_id = tus_uploader.get_file_id()
        if file_id is None:
            raise RuntimeError("Failed to get file id for project {project_name}.")
        return file_id

    def export_project(
        self,
        project_id: str,
        filepath: os.PathLike,
        include_models: str | IncludeModelsType = IncludeModelsType.ALL,
    ):
        """
        Export a project from the Geti Platform.

        :param project_id: The ID of the project to export.
        :param filepath: The path to save the exported project.
        :param include_models: Indicates which models to include, supported are 'all', 'none' or 'latest_active'
        :raises: RuntimeError if the download url is not retrieved.
        """
        if isinstance(include_models, str):
            include_models = IncludeModelsType(include_models)
        url = f"{self.base_url}projects/{project_id}:export?include_models={include_models}"
        self._export_snapshot(url=url, filepath=filepath)

    def export_dataset(
        self,
        project: Project,
        dataset: Dataset,
        filepath: os.PathLike,
        export_format: DatasetFormat = DatasetFormat.DATUMARO,
        include_unannotated_media: bool = False,
    ):
        """
        Export a dataset from the Geti Platform.

        :param project: The project containing the dataset.
        :param dataset: The dataset to export.
        :param filepath: The path to save the exported dataset.
        :param export_format: The format to export the dataset in.
        :param include_unannotated_media: Whether to include media that has not been annotated.
        :raises: RuntimeError if the download url is not retrieved.
        """
        query_params = (
            f"export_format={str(export_format)}&include_unannotated_media={str(include_unannotated_media).lower()}"
        )
        url = f"{self.base_url}projects/{project.id}/datasets/{dataset.id}:prepare-for-export?{query_params}"

        self._export_snapshot(url=url, filepath=filepath)

    def _export_snapshot(self, url: str, filepath: os.PathLike):
        """
        Export an entity from the Geti Platform.

        :param url: The export endpoint.
        :param filepath: The path to save the exported entity.
        :raises: RuntimeError if the download url is not retrieved.
        """
        parent_dir = os.path.dirname(filepath)
        os.makedirs(parent_dir, exist_ok=True)

        response = self.session.get_rest_response(
            url=url,
            method="POST",
        )
        if response.get("job_id") is None:
            raise RuntimeError("Failed to get job id for the export entity.")

        job = get_job_with_timeout(
            job_id=response.get("job_id"),
            session=self.session,
            workspace_id=self.workspace_id,
            job_type="export_project",
        )

        job = monitor_job(session=self.session, job=job, interval=5)
        if job.metadata.download_url is None:
            raise RuntimeError("Failed to get download url for the exported entity.")
        url = job.metadata.download_url

        if not url.startswith("/"):
            url = "/" + url

        logging.info("Downloading the archive...")
        zip_response = self.session.get_rest_response(url=url, method="GET", contenttype="multipart")
        with open(filepath, "wb") as f:
            f.write(zip_response.content)
