# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
import logging
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from vcr import VCR

from geti_sdk import Geti
from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.data_models import Project, TaskType
from geti_sdk.rest_clients import (
    AnnotationClient,
    ConfigurationClient,
    DatasetClient,
    ImageClient,
    ModelClient,
    PredictionClient,
    ProjectClient,
    ProjectConfigurationClient,
    TrainingClient,
    TrainingConfigurationClient,
    VideoClient,
)
from tests.helpers import DatumAnnotationReader

from .constants import CASSETTE_EXTENSION, PROJECT_PREFIX
from .finalizers import force_delete_project


class ProjectService:
    """
    This class contains functionality to quickly create projects and interact with
    them through the respective clients

    :param geti: Geti instance representing the GETi server and workspace in
        which to create the project
    :param vcr: VCR instance used for recording HTTP requests made during the project
        lifespan. If left as None, no requests will be recorded or played back from
        VCR cassettes
    """

    def __init__(self, geti: Geti, vcr: VCR | None = None, is_offline: bool = False):
        if vcr is None:
            self.vcr_context = nullcontext
        else:
            self.vcr_context = vcr.use_cassette
        self.session = geti.session
        self.workspace_id = geti.workspace_id
        self.geti = geti
        self.project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)

        self._project: Project | None = None
        self._project_creation_timestamp: float | None = None
        self._is_offline: bool = is_offline
        self._configuration_client: ConfigurationClient | None = None
        self._project_configuration_client: ConfigurationClient | None = None
        self._training_configuration_client: ConfigurationClient | None = None
        self._image_client: ImageClient | None = None
        self._annotation_client: AnnotationClient | None = None
        self._training_client: TrainingClient | None = None
        self._video_client: VideoClient | None = None
        self._model_client: ModelClient | None = None
        self._prediction_client: PredictionClient | None = None
        self._dataset_client: DatasetClient | None = None
        self._client_names = [
            "_configuration_client",
            "_image_client",
            "_annotation_client",
            "_training_client",
            "_video_client",
            "_model_client",
            "_prediction_client",
            "_dataset_client",
        ]
        self._project_removal_delay = 5  # seconds

    def create_project(
        self,
        project_name: str | None = None,
        project_type: str = "classification",
        labels: list[list[str] | list[dict[str, Any]]] | None = None,
        keypoint_structure: dict[str, list] | None = None,
    ) -> Project:
        """
        Create a project according to the `name`, `project_type` and `labels` specified.

        :param project_name: Name of the project to create
        :param project_type: Type of the project to create
        :param labels: List of labels for each tas
        :param keypoint_structure: The structure of the keypoints to be used for the project,
            represented as a graph of nodes and edges.
        :return: the created project
        """
        if project_name is None:
            project_name = f"{PROJECT_PREFIX}_project_simple"
        if self._project is None:
            if labels is None:
                labels = [["cube", "cylinder"]]
            with self.vcr_context(f"{project_name}.{CASSETTE_EXTENSION}"):
                project = self.project_client.create_project(
                    project_name=project_name,
                    project_type=project_type,
                    labels=labels,
                    keypoint_structure=keypoint_structure,
                )
                self.project = project
                return project
        else:
            raise ValueError(
                "This ProjectService instance already contains an existing project. "
                "Please either delete the existing project first or use a new "
                "instance to create another project"
            )

    def get_or_create_project(
        self,
        project_name: str = "sdk_test_project_simple",
        project_type: str = "classification",
        labels: list[list[str] | list[dict[str, Any]]] | None = None,
        keypoint_structure: dict[str, list] | None = None,
    ) -> Project:
        """
        This method will always return a project. It will either create a new one, or
        return the existing project if it has already been created.

        :param project_name: Name of the project to create
        :param project_type: Type of the project to create
        :param labels: List of labels for each task
        :param keypoint_structure: The structure of the keypoints to be used for the project,
            represented as a graph of nodes and edges.
        :return: the existing or newly created project
        """
        if not self.has_project:
            self.create_project(
                project_name=project_name,
                project_type=project_type,
                labels=labels,
                keypoint_structure=keypoint_structure,
            )
        return self.project

    def create_project_from_dataset(
        self,
        annotation_readers: list[AnnotationReader],
        project_name: str = "sdk_test_project_simple",
        project_type: str = "classification",
        path_to_dataset: str = "",
        n_images: int = 12,
    ):
        """
        Create a project from a dataset, and upload media and annotations to it

        :param annotation_readers: List of annotation readers that should be used as
            annotation sources for the tasks in the project
        :param project_name: Name of the project to create
        :param project_type: Type of the project to create
        :param path_to_dataset: Path to the base folder containing the dataset
        :param n_images: Number of annotated images to upload. If set to -1, uploads
            all images for which the annotation reader contains annotations
        :raises: ValueError in case the ProjectService already contains a project
        :return: The Project that was created
        """
        if not self.has_project:
            with self.vcr_context(f"{project_name}.{CASSETTE_EXTENSION}"):
                if len(annotation_readers) == 1:
                    annotation_reader = annotation_readers[0]
                    project = self.geti.create_single_task_project_from_dataset(
                        project_name=project_name,
                        project_type=project_type,
                        path_to_images=path_to_dataset,
                        annotation_reader=annotation_reader,
                        number_of_images_to_upload=n_images,
                        number_of_images_to_annotate=n_images,
                        enable_auto_train=False,
                        upload_videos=True,
                        max_threads=1,
                    )
                else:
                    project = self.geti.create_task_chain_project_from_dataset(
                        project_name=project_name,
                        project_type=project_type,
                        path_to_images=path_to_dataset,
                        label_source_per_task=annotation_readers,
                        number_of_images_to_upload=n_images,
                        number_of_images_to_annotate=n_images,
                        enable_auto_train=False,
                        max_threads=1,
                    )
        else:
            raise ValueError(
                "This ProjectService instance already contains an existing project. "
                "Please either delete the existing project first or use a new "
                "instance to create another project"
            )
        self.project = project
        return project

    @property
    def project(self) -> Project:
        """
        Returns the project managed by the ProjectService.

        :return: The project managed by the ProjectService
        :raises: ValueError if the ProjectService does not contain a project yet
        """
        if self._project is None:
            raise ValueError(
                "This ProjectService instance does not contain a project yet. Please "
                "call `ProjectService.create_project` to create a new project first."
            )
        return self._project

    @project.setter
    def project(self, value: Project | None) -> None:
        """
        Set the project for the ProjectService.
        """
        self._project = value
        if self._project is None:
            self._project_creation_timestamp = None
        else:
            self._project_creation_timestamp = time.time()

    @property
    def has_project(self) -> bool:
        """
        Returns True if the ProjectService contains an existing project, False otherwise
        """
        return self._project is not None

    @property
    def is_task_chain_project(self) -> bool:
        """
        Returns True if the project belonging to this ProjectService instance is a task
        chain project, False otherwise

        If this method is called while no project is defined yet, a ValueError will be
        raised
        """
        return len(self.project.get_trainable_tasks()) > 1

    @property
    def image_client(self) -> ImageClient:
        """Returns the ImageClient instance for the project"""
        if self._image_client is None:
            with self.vcr_context(f"{self.project.name}_image_client.{CASSETTE_EXTENSION}"):
                self._image_client = ImageClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._image_client

    @property
    def video_client(self) -> VideoClient:
        """Returns the VideoClient instance for the project"""
        if self._video_client is None:
            with self.vcr_context(f"{self.project.name}_video_client.{CASSETTE_EXTENSION}"):
                self._video_client = VideoClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._video_client

    @property
    def annotation_client(self) -> AnnotationClient:
        """Returns the AnnotationClient instance for the project"""
        if self._annotation_client is None:
            with self.vcr_context(f"{self.project.name}_annotation_client.{CASSETTE_EXTENSION}"):
                self._annotation_client = AnnotationClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._annotation_client

    @property
    def dataset_client(self) -> DatasetClient:
        """Returns the DatasetClient instance for the project"""
        if self._dataset_client is None:
            with self.vcr_context(f"{self.project.name}_dataset_client.{CASSETTE_EXTENSION}"):
                self._dataset_client = DatasetClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._dataset_client

    @property
    def configuration_client(self) -> ConfigurationClient:
        """Returns the ConfigurationClient instance for the project"""
        if self._configuration_client is None:
            with self.vcr_context(f"../LEGACY/{self.project.name}_configuration_client.{CASSETTE_EXTENSION}"):
                self._configuration_client = ConfigurationClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._configuration_client

    @property
    def project_configuration_client(self) -> ProjectConfigurationClient:
        """Returns the ConfigurationClient instance for the project"""
        if self._project_configuration_client is None:
            with self.vcr_context(f"{self.project.name}_project_configuration_client.{CASSETTE_EXTENSION}"):
                self._project_configuration_client = ProjectConfigurationClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._project_configuration_client

    @property
    def training_configuration_client(self) -> TrainingConfigurationClient:
        """Returns the ConfigurationClient instance for the project"""
        if self._training_configuration_client is None:
            with self.vcr_context(f"{self.project.name}_training_configuration_client.{CASSETTE_EXTENSION}"):
                self._training_configuration_client = TrainingConfigurationClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._training_configuration_client

    @property
    def training_client(self) -> TrainingClient:
        """Returns the TrainingClient instance for the project"""
        if self._training_client is None:
            with self.vcr_context(f"{self.project.name}_training_client.{CASSETTE_EXTENSION}"):
                self._training_client = TrainingClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._training_client

    @property
    def prediction_client(self) -> PredictionClient:
        """Returns the PredictionClient instance for the project"""
        if self._prediction_client is None:
            with self.vcr_context(f"{self.project.name}_prediction_client.{CASSETTE_EXTENSION}"):
                self._prediction_client = PredictionClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._prediction_client

    @property
    def model_client(self) -> ModelClient:
        """Returns the ModelClient instance for the project"""
        if self._model_client is None:
            with self.vcr_context(f"{self.project.name}_model_client.{CASSETTE_EXTENSION}"):
                self._model_client = ModelClient(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project,
                )
        return self._model_client

    @property
    def is_training(self) -> bool:
        """
        Returns True if the project service contains a project, and that project is
        currently running at least one training job for any of its tasks. Returns
        False otherwise.

        :return: True if the project is training, False otherwise
        """
        if not self.has_project:
            return False
        with self.vcr_context(f"{self.project.name}_is_training.{CASSETTE_EXTENSION}"):
            return self.training_client.is_training()

    def delete_project(self):
        """Deletes the project from the server"""
        if self.has_project:
            with self.vcr_context(f"{self.project.name}_deletion.{CASSETTE_EXTENSION}"):
                # Server needs a moment to process the project before deletion
                if (
                    not self._is_offline
                    and (lifetime := time.time() - self._project_creation_timestamp) < self._project_removal_delay
                ):
                    time.sleep(self._project_removal_delay - lifetime)
                force_delete_project(self.project, self.project_client)
                self.reset_state()

    def reset_state(self) -> None:
        """
        Resets the state of the ProjectService instance. This method should be called
        once the project belonging to the project service is deleted from the server
        """
        self.project = None
        for client_name in self._client_names:
            setattr(self, client_name, None)

    def add_annotated_media(self, annotation_readers: Sequence[AnnotationReader], n_images: int = 12) -> None:
        """
        Adds annotated media to the project

        :param annotation_readers: List of AnnotationReader instances to use to get
            annotations for the media. The length of this list must match the number
            of trainable tasks in the project
        :param n_images: Number of annotated images to upload. If set to -1, uploads
            all images for which the annotation reader contains annotations
        """
        annotation_reader_1 = annotation_readers[0]
        data_path = annotation_reader_1.base_folder

        if len(annotation_readers) != len(self.project.get_trainable_tasks()):
            raise ValueError(
                "Number of annotation readers received does not match the number of "
                "trainable tasks in the project: Unable to upload annotated media"
            )
        with self.vcr_context(f"{self.project.name}_add_annotated_media.{CASSETTE_EXTENSION}"):
            if isinstance(annotation_reader_1, DatumAnnotationReader):
                images = self.image_client.upload_from_list(
                    path_to_folder=data_path,
                    image_names=annotation_reader_1.get_all_image_names(),
                    n_images=n_images,
                    max_threads=1,
                )
            else:
                images = self.image_client.upload_folder(data_path, n_images=n_images, max_threads=1)

            if n_images < len(images) and n_images != -1:
                images = images[:n_images]
            # Annotation preparation and upload
            for task_index, task in enumerate(self.project.get_trainable_tasks()):
                # Set annotation reader task type
                annotation_readers[task_index].task_type = task.type
                annotation_readers[task_index].prepare_and_set_dataset(task_type=task.type)
                self.annotation_client.annotation_reader = annotation_readers[task_index]
                # Upload annotations
                self.annotation_client.upload_annotations_for_images(
                    images=images, append_annotations=task_index > 0, max_threads=1
                )

    def set_auto_train(self, auto_train: bool = True) -> None:
        """
        Sets the 'auto_training' parameter for all tasks in the project to `auto_train`

        :param auto_train: True to turn auto_training on, False to turn it off
        """
        with self.vcr_context(f"{self.project.name}_set_auto_train.{CASSETTE_EXTENSION}"):
            if self.session.version.is_configuration_revamped:
                self.project_configuration_client.set_project_auto_train(auto_train=auto_train)
            else:
                self.configuration_client.set_project_auto_train(auto_train=auto_train)

    def set_minimal_training_hypers(self) -> None:
        """
        Configures the project to use the lowest possible batch size and number of
        epochs to perform a minimal training round

        """
        with self.vcr_context(f"{self.project.name}_set_minimal_hypers.{CASSETTE_EXTENSION}"):
            self.configuration_client.set_project_num_iterations(1)
            for task in self.project.get_trainable_tasks():
                task_config = self.configuration_client.get_task_configuration(task.id)
                try:
                    task_config.set_parameter_value("batch_size", 1)
                    self.configuration_client.set_configuration(task_config)
                except ValueError:
                    logging.warning(
                        f"Parameter batch_size was not found in the configuration for "
                        f"task {task.summary}. Unable to configure batch size"
                    )

    def set_reduced_memory_hypers(self) -> None:
        """
        Reduce batch size in memory intensive tasks to avoid OOM errors in pods. Use
        default hypers for other tasks
        """
        with self.vcr_context(f"{self.project.name}_set_reduced_memory_hypers.{CASSETTE_EXTENSION}"):
            for task in self.project.get_trainable_tasks():
                if task.type in [
                    TaskType.DETECTION,
                    TaskType.ROTATED_DETECTION,
                    TaskType.INSTANCE_SEGMENTATION,
                ]:
                    task_hypers = self.configuration_client.get_task_configuration(task_id=task.id)
                    task_hypers.batch_size.value = 1
                    self.configuration_client.set_configuration(task_hypers)

    def set_auto_training_annotation_requirement(self, required_images: int = 6) -> None:
        """
        Sets the 'Number of images required for auto-training' parameter for all
        tasks in the project to `required_images`

        :param required_images: Number of images required before starting a new round
            of auto training for the task
        """
        with self.vcr_context(f"{self.project.name}_set_auto_training_annotation_requirement.{CASSETTE_EXTENSION}"):
            self.configuration_client.set_project_parameter(
                parameter_name="required_images_auto_training", value=required_images
            )
