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
import os
import time
from typing import List

import cv2
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from vcr import VCR

from geti_sdk import Geti
from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.data_models import Job, Prediction, Project
from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiRequestException
from geti_sdk.post_inference_hooks import (
    AlwaysTrigger,
    ConfidenceTrigger,
    FileSystemDataCollection,
    GetiDataCollection,
    PostInferenceHook,
)
from geti_sdk.rest_clients import (
    AnnotationClient,
    DatasetClient,
    ImageClient,
    VideoClient,
)
from geti_sdk.utils import show_video_frames_with_annotation_scenes
from tests.helpers import (
    ProjectService,
    SdkTestMode,
    attempt_to_train_task,
    await_training_start,
    get_or_create_annotated_project_for_test_class,
)
from tests.helpers.constants import CASSETTE_EXTENSION, PROJECT_PREFIX


class TestGeti:
    """
    Integration tests for the methods in the Geti class.

    NOTE: These tests are meant to be run in one go
    """

    @staticmethod
    def ensure_annotated_project(
        project_service: ProjectService,
        annotation_readers: List[AnnotationReader],
        project_type: str,
        use_create_from_dataset: bool = False,
        path_to_dataset: str = "",
    ) -> Project:
        project_name = f"{PROJECT_PREFIX}_geti_{project_type}"

        if not use_create_from_dataset:
            return get_or_create_annotated_project_for_test_class(
                project_service=project_service,
                annotation_readers=annotation_readers,
                project_type=project_type,
                project_name=project_name,
                enable_auto_train=False,
            )
        else:
            return project_service.create_project_from_dataset(
                annotation_readers=annotation_readers,
                project_name=project_name,
                project_type=project_type,
                path_to_dataset=path_to_dataset,
                n_images=-1,
            )

    @pytest.mark.parametrize(
        "project_service, project_type, annotation_readers, use_create_from_dataset, path_to_media",
        [
            (
                "fxt_project_service",
                "classification",
                "fxt_geti_annotation_reader",
                True,
                "fxt_light_bulbs_dataset",
            ),
            (
                "fxt_project_service_2",
                "detection_to_classification",
                "fxt_classification_to_detection_annotation_readers",
                False,
                "fxt_blocks_dataset",
            ),
        ],
        ids=["Single task project", "Task chain project"],
    )
    def test_project_setup(
        self,
        project_service,
        project_type,
        annotation_readers,
        use_create_from_dataset,
        path_to_media,
        request: FixtureRequest,
        fxt_vcr: VCR,
        fxt_test_mode: SdkTestMode,
    ):
        """
        This test sets up an annotated project on the server, that persists for the
        duration of this test class. The project will train while the project
        creation tests are running.
        """
        lazy_fxt_project_service = request.getfixturevalue(project_service)
        lazy_fxt_annotation_reader = request.getfixturevalue(annotation_readers)
        lazy_fxt_dataset_path = request.getfixturevalue(path_to_media)

        if not isinstance(lazy_fxt_annotation_reader, list):
            lazy_fxt_annotation_reader = [lazy_fxt_annotation_reader]

        project = self.ensure_annotated_project(
            project_service=lazy_fxt_project_service,
            annotation_readers=lazy_fxt_annotation_reader,
            project_type=project_type,
            use_create_from_dataset=use_create_from_dataset,
            path_to_dataset=lazy_fxt_dataset_path,
        )
        assert lazy_fxt_project_service.has_project

        # For the integration tests we start training manually
        with fxt_vcr.use_cassette(
            f"{project.name}_setup_training.{CASSETTE_EXTENSION}"
        ):
            jobs: List[Job] = []
            for task in project.get_trainable_tasks():
                jobs.append(
                    attempt_to_train_task(
                        training_client=lazy_fxt_project_service.training_client,
                        task=task,
                        test_mode=fxt_test_mode,
                    )
                )

        await_training_start(fxt_test_mode, lazy_fxt_project_service.training_client)

        assert lazy_fxt_project_service.is_training

    def test_geti_initialization(self, fxt_geti: Geti):
        """
        Test that the Geti instance is initialized properly, by checking that it
        obtains a workspace ID
        """
        assert fxt_geti.workspace_id is not None

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_type, dataset_filter_criterion, annotation_reader, default_labels, image_folder",
        [
            # (
            #     "classification",
            #     "XOR",
            #     "fxt_annotation_reader",
            #     "fxt_default_labels",
            #     "fxt_image_folder",
            # ),
            (
                "detection",
                "OR",
                "fxt_annotation_reader",
                "fxt_default_labels",
                "fxt_image_folder",
            ),
            (
                "segmentation",
                "OR",
                "fxt_annotation_reader",
                "fxt_default_labels",
                "fxt_image_folder",
            ),
            (
                "instance_segmentation",
                "OR",
                "fxt_annotation_reader",
                "fxt_default_labels",
                "fxt_image_folder",
            ),
            (
                "rotated_detection",
                "OR",
                "fxt_annotation_reader",
                "fxt_default_labels",
                "fxt_image_folder",
            ),
            # (
            #     "keypoint_detection",
            #     "OR",
            #     "fxt_annotation_reader_keypoint",
            #     "fxt_default_keypoint_labels",
            #     "fxt_human_pose_image_folder",
            # ),
        ],
        ids=[
            # "Classification project",  # TODO re-enable after fixing dataset
            "Detection project",
            "Segmentation project",
            "Instance segmentation project",
            "Rotated detection project",
            # "Keypoint detection project",  # TODO re-enable after fixing dataset
        ],
    )
    def test_create_single_task_project_from_dataset(
        self,
        project_type,
        dataset_filter_criterion,
        annotation_reader: str,
        fxt_geti: Geti,
        default_labels: str,
        image_folder: str,
        fxt_project_finalizer,
        request: FixtureRequest,
    ):
        """
        Test that creating a single task project from a datumaro dataset works

        Tests project creation for classification, detection and segmentation type
        projects
        """
        annotation_reader = request.getfixturevalue(annotation_reader)
        default_labels = request.getfixturevalue(default_labels)
        image_folder = request.getfixturevalue(image_folder)
        project_name = f"{PROJECT_PREFIX}_{project_type}_project_from_dataset"
        annotation_reader.filter_dataset(
            labels=default_labels, criterion=dataset_filter_criterion
        )
        keypoint_structure = None
        if project_type == "keypoint_detection":
            joints = annotation_reader.get_keypoint_joints()
            edges = [
                {"nodes": [default_labels[a - 1], default_labels[b - 1]]}
                for a, b in joints
            ]
            positions = [
                {"label": label, "x": np.random.random(), "y": np.random.random()}
                for label in default_labels
            ]
            keypoint_structure = {"edges": edges, "positions": positions}
        project = fxt_geti.create_single_task_project_from_dataset(
            project_name=project_name,
            project_type=project_type,
            path_to_images=image_folder,
            annotation_reader=annotation_reader,
            enable_auto_train=False,
            max_threads=1,
            keypoint_structure=keypoint_structure,
        )

        request.addfinalizer(lambda: fxt_project_finalizer(project))

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_type",
        ["detection_to_classification", "detection_to_segmentation"],
        ids=[
            "Detection to classification project",
            "Detection to segmentation project",
        ],
    )
    def test_create_task_chain_project_from_dataset(
        self,
        project_type,
        fxt_annotation_reader_factory,
        fxt_geti: Geti,
        fxt_default_labels: List[str],
        fxt_image_folder: str,
        fxt_project_finalizer,
        request: FixtureRequest,
    ):
        """
        Test that creating a task chain project from a datumaro dataset works

        Tests project creation for:
          detection -> classification
          detection -> segmentation
        """
        project_name = f"{PROJECT_PREFIX}_{project_type}_project_from_dataset"
        annotation_reader_task_1 = fxt_annotation_reader_factory()
        annotation_reader_task_2 = fxt_annotation_reader_factory()
        annotation_reader_task_1.filter_dataset(
            labels=fxt_default_labels, criterion="OR"
        )
        annotation_reader_task_2.filter_dataset(
            labels=fxt_default_labels, criterion="OR"
        )
        annotation_reader_task_1.group_labels(
            labels_to_group=fxt_default_labels, group_name="block"
        )
        project = fxt_geti.create_task_chain_project_from_dataset(
            project_name=project_name,
            project_type=project_type,
            path_to_images=fxt_image_folder,
            label_source_per_task=[annotation_reader_task_1, annotation_reader_task_2],
            enable_auto_train=False,
            max_threads=1,
        )
        request.addfinalizer(lambda: fxt_project_finalizer(project))

        all_labels = fxt_default_labels + ["block"]
        for label_name in all_labels:
            assert label_name in [label.name for label in project.get_all_labels()]

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_service, include_videos",
        [("fxt_project_service", True), ("fxt_project_service_2", False)],
        ids=["Single task project", "Task chain project"],
    )
    def test_download_and_upload_project(
        self,
        project_service,
        include_videos,
        fxt_geti: Geti,
        fxt_temp_directory: str,
        fxt_project_finalizer,
        request: FixtureRequest,
    ):
        """
        Test that downloading a project works as expected.

        """
        lazy_fxt_project_service = request.getfixturevalue(project_service)
        project = lazy_fxt_project_service.project
        target_folder = os.path.join(fxt_temp_directory, project.name)

        fxt_geti.download_project_data(
            project,
            target_folder=target_folder,
            max_threads=1,
        )

        assert os.path.isdir(target_folder)
        assert "project.json" in os.listdir(target_folder)

        n_images = len(os.listdir(os.path.join(target_folder, "images", "Dataset")))
        n_annotations = len(os.listdir(os.path.join(target_folder, "annotations")))

        uploaded_project = fxt_geti.upload_project_data(
            target_folder=target_folder,
            project_name=f"{project.name}_upload",
            enable_auto_train=False,
            max_threads=1,
        )
        request.addfinalizer(lambda: fxt_project_finalizer(uploaded_project))
        image_client = ImageClient(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=uploaded_project,
        )
        images = image_client.get_all_images()
        assert len(images) == n_images

        annotation_client = AnnotationClient(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=uploaded_project,
        )
        annotation_target_folder = os.path.join(
            fxt_temp_directory, "uploaded_annotations", project.name
        )

        if include_videos:
            video_client = VideoClient(
                session=fxt_geti.session,
                workspace_id=fxt_geti.workspace_id,
                project=uploaded_project,
            )
            n_videos = len(os.listdir(os.path.join(target_folder, "videos", "Dataset")))
            videos = video_client.get_all_videos()

            assert len(videos) == n_videos
            annotation_client.download_all_annotations(
                annotation_target_folder, max_threads=1
            )

        else:
            annotation_client.download_annotations_for_images(
                images, annotation_target_folder, max_threads=1
            )

        assert (
            len(os.listdir(os.path.join(annotation_target_folder, "annotations")))
            == n_annotations
        )

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_service",
        ["fxt_project_service", "fxt_project_service_2"],
        ids=["Single task project", "Task chain project"],
    )
    def test_upload_and_predict_image(
        self,
        project_service,
        request: FixtureRequest,
        fxt_geti: Geti,
        fxt_image_path: str,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Verifies that the upload_and_predict_image method works correctly
        """
        lazy_fxt_project_service = request.getfixturevalue(project_service)
        project = lazy_fxt_project_service.project

        # If training is not ready yet, monitor progress until job completes
        if fxt_test_mode != SdkTestMode.OFFLINE:
            timeout = 300
            t_start = time.time()
            training = lazy_fxt_project_service.training_client.is_training()
            while training and time.time() - t_start < timeout:
                training = lazy_fxt_project_service.training_client.is_training()
                time.sleep(10)

        # Create a 'test' dataset in the project
        dataset_client = DatasetClient(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=project,
        )
        test_dataset_name = "test_dataset"
        dataset_client.create_dataset(name=test_dataset_name)

        # Make several attempts to get the prediction, first attempts trigger the
        # inference server to start up but the requests may time out
        n_attempts = 2 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        sleep_time = 20 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        for j in range(n_attempts):
            try:
                image, prediction = fxt_geti.upload_and_predict_image(
                    project=project,
                    image=fxt_image_path,
                    visualise_output=False,
                    delete_after_prediction=False,
                    dataset_name=test_dataset_name,
                )
            except GetiRequestException as error:
                prediction = None
                time.sleep(sleep_time)
                logging.info(error)
            if prediction is not None:
                assert isinstance(prediction, Prediction)
                break

    @pytest.mark.vcr()
    def test_upload_and_predict_video(
        self,
        fxt_project_service: ProjectService,
        fxt_geti: Geti,
        fxt_video_path_1_light_bulbs: str,
        fxt_temp_directory: str,
    ) -> None:
        """
        Verify that the `Geti.upload_and_predict_video` method works as expected
        """
        video, frames, predictions = fxt_geti.upload_and_predict_video(
            project=fxt_project_service.project,
            video=fxt_video_path_1_light_bulbs,
            visualise_output=False,
        )
        assert len(frames) == len(predictions)
        video_filepath = os.path.join(fxt_temp_directory, "inferred_video.avi")
        show_video_frames_with_annotation_scenes(
            video_frames=frames, annotation_scenes=predictions, filepath=video_filepath
        )
        assert os.path.isfile(video_filepath)

        # Check that invalid project raises a KeyError
        with pytest.raises(KeyError):
            project = fxt_geti.get_project(project_name="invalid_project_name")
            fxt_geti.upload_and_predict_video(
                project=project,
                video=fxt_video_path_1_light_bulbs,
                visualise_output=False,
            )

        # Check that video is not uploaded if it's already in the project
        video, frames, predictions = fxt_geti.upload_and_predict_video(
            project=fxt_project_service.project,
            video=video,
            visualise_output=False,
        )
        assert len(frames) == len(predictions)

        # Check that uploading list of numpy arrays as video works
        new_frames = video.to_frames(frame_stride=50, include_data=True)
        np_frames = [frame.numpy for frame in new_frames]
        np_video, frames, predictions = fxt_geti.upload_and_predict_video(
            project=fxt_project_service.project,
            video=np_frames,
            visualise_output=False,
            delete_after_prediction=True,
        )
        assert len(frames) == len(predictions)
        videos = fxt_project_service.video_client.get_all_videos()
        assert np_video.id not in [vid.id for vid in videos]

    @pytest.mark.vcr()
    def test_upload_and_predict_media_folder(
        self,
        fxt_project_service: ProjectService,
        fxt_geti: Geti,
        fxt_video_folder_light_bulbs: str,
        fxt_image_folder_light_bulbs: str,
        fxt_temp_directory: str,
    ) -> None:
        """
        Verify that the `Geti.upload_and_predict_media_folder` method works as expected
        """
        video_output_folder = os.path.join(fxt_temp_directory, "inferred_videos")
        image_output_folder = os.path.join(fxt_temp_directory, "inferred_images")

        video_success = fxt_geti.upload_and_predict_media_folder(
            project=fxt_project_service.project,
            media_folder=fxt_video_folder_light_bulbs,
            output_folder=video_output_folder,
            delete_after_prediction=True,
            max_threads=1,
        )
        image_success = fxt_geti.upload_and_predict_media_folder(
            project=fxt_project_service.project,
            media_folder=fxt_image_folder_light_bulbs,
            output_folder=image_output_folder,
            delete_after_prediction=True,
            max_threads=1,
        )

        assert video_success
        assert image_success

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_service",
        ["fxt_project_service", "fxt_project_service_2"],
        ids=["Single task project", "Task chain project"],
    )
    def test_deployment(
        self,
        project_service,
        request: FixtureRequest,
        fxt_geti: Geti,
        fxt_image_path: str,
        fxt_temp_directory: str,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Verifies that deploying a project works
        """
        lazy_fxt_project_service = request.getfixturevalue(project_service)
        project = lazy_fxt_project_service.project
        deployment_folder = os.path.join(fxt_temp_directory, project.name)

        n_attempts = 2 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        sleep_time = 20 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        for _ in range(n_attempts):
            try:
                deployment = fxt_geti.deploy_project(
                    project=project,
                    output_folder=deployment_folder,
                    enable_explainable_ai=True,
                )
            except (ValueError, FileNotFoundError) as error:
                deployment = None
                time.sleep(sleep_time)
                logging.info(error)
        assert deployment is not None

        assert os.path.isdir(os.path.join(deployment_folder, "deployment"))
        deployment.load_inference_models(device="CPU")

        image_bgr = cv2.imread(fxt_image_path)
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        local_prediction = deployment.infer(image_np)
        assert isinstance(local_prediction, Prediction)
        image, online_prediction = fxt_geti.upload_and_predict_image(
            project,
            image=image_np,
            delete_after_prediction=True,
            visualise_output=False,
        )

        explain_prediction = deployment.explain(image_np)

        if all([model.has_xai_head for model in deployment.models]):
            assert explain_prediction.feature_vector is not None

        assert len(explain_prediction.maps) > 0

        online_mask = online_prediction.as_mask(image.media_information)
        local_mask = local_prediction.as_mask(image.media_information)

        assert online_mask.shape == local_mask.shape
        # assert np.all(local_mask == online_mask)

        deployment_from_folder = Deployment.from_folder(
            path_to_folder=deployment_folder
        )
        assert deployment_from_folder.models[0].name == deployment.models[0].name

    @pytest.mark.vcr()
    def test_post_inference_hooks(
        self,
        fxt_project_service: ProjectService,
        fxt_geti: Geti,
        fxt_image_path: str,
        fxt_temp_directory: str,
    ):
        """
        Test that adding post inference hooks to a deployment works, and that the
        hooks function as expected
        """
        project = fxt_project_service.project
        deployment_folder = os.path.join(fxt_temp_directory, project.name)

        deployment = fxt_geti.deploy_project(project=project)
        dataset_name = "Test hooks"

        # Add a GetiDataCollectionHook
        trigger = AlwaysTrigger()
        action = GetiDataCollection(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=project,
            dataset=dataset_name,
        )
        hook = PostInferenceHook(trigger=trigger, action=action)
        deployment.add_post_inference_hook(hook)

        # Add a FileSystemDataCollection hook
        hook_data = os.path.join(deployment_folder, "hook_data")
        trigger_2 = ConfidenceTrigger(threshold=1.1)
        action_2 = FileSystemDataCollection(target_folder=hook_data)
        hook_2 = PostInferenceHook(trigger=trigger_2, action=action_2, max_threads=0)
        deployment.add_post_inference_hook(hook_2)

        deployment.load_inference_models(device="CPU")
        image_bgr = cv2.imread(fxt_image_path)
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _ = deployment.infer(image_np)

        # Small delay to ensure that the hooks have time to run
        time.sleep(1)

        dataset_client = DatasetClient(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=project,
        )

        assert len(deployment.post_inference_hooks) == 2

        # Assert that the hooks have fired
        dataset = dataset_client.get_dataset_by_name(dataset_name)
        hook_images = fxt_project_service.image_client.get_all_images(dataset=dataset)
        assert len(hook_images) == 1
        expected_folders = ["images", "overlays", "predictions", "scores"]
        for folder_name in expected_folders:
            assert folder_name in os.listdir(hook_data)
            assert len(os.listdir(os.path.join(hook_data, folder_name))) == 1

        # Set deployment to async mode
        results: List[Prediction] = []

        def process_results(image, prediction, data):
            results.append(prediction)

        deployment.set_asynchronous_callback(process_results)
        assert deployment.asynchronous_mode

        deployment.infer_async(image_np)
        deployment.await_all()

        # Assert that the process_results callback has run
        assert len(results) == 1

        # Small delay to ensure that the hooks have time to run
        time.sleep(1)

        # Assert that the hooks have fired in the async case: 1 image should have
        # been added to both Geti dataset and results folders
        hook_images = fxt_project_service.image_client.get_all_images(dataset=dataset)
        assert len(hook_images) == 2
        for folder_name in expected_folders:
            assert len(os.listdir(os.path.join(hook_data, folder_name))) == 2

        deployment.clear_inference_hooks()
        assert len(deployment.post_inference_hooks) == 0

        deployment.asynchronous_mode = False
        assert not deployment._async_callback_defined

    @pytest.mark.vcr()
    def test_download_project_including_models_and_predictions(
        self,
        fxt_project_service: ProjectService,
        fxt_geti: Geti,
        fxt_temp_directory: str,
    ):
        """
        Test that downloading a project including predictions, active models and
        deployment works as expected.
        """
        project = fxt_project_service.project
        target_folder = os.path.join(
            fxt_temp_directory, project.name + "_all_inclusive"
        )
        fxt_geti.download_project_data(
            project=project,
            target_folder=target_folder,
            include_predictions=True,
            include_active_models=True,
            include_deployment=True,
            max_threads=1,
        )

        prediction_folder_name = "predictions"
        deployment_folder_name = "deployment"
        model_folder_name = "models"

        prediction_path = os.path.join(target_folder, prediction_folder_name)
        deployment_path = os.path.join(target_folder, deployment_folder_name)
        model_path = os.path.join(target_folder, model_folder_name)

        assert os.path.isdir(prediction_path)
        assert os.path.isdir(deployment_path)
        assert os.path.isdir(model_path)

        # Check the contents of the downloaded predictions
        assert os.path.isdir(os.path.join(prediction_path, "saliency_maps"))

        # Check the downloaded deployment
        deployment = Deployment.from_folder(deployment_path)
        assert deployment.project.name == project.name
        assert len(deployment.models) == len(project.get_trainable_tasks())

        # Check the contents of the downloaded active model folder
        model_contents = os.listdir(model_path)
        assert (
            f"{project.get_trainable_tasks()[0].type}_model_details.json"
            in model_contents
        )

        found_base_model = False
        found_optimized_model = False
        for filename in model_contents:
            if "_base.zip" in filename:
                found_base_model = True
            if "_optimized.zip" in filename:
                found_optimized_model = True
        assert found_optimized_model and found_base_model
