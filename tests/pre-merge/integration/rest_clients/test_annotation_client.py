import copy
import json
import os
import time
from typing import List

import pytest

from geti_sdk.annotation_readers import GetiAnnotationReader
from geti_sdk.data_models import AnnotationScene, Project, Video, VideoFrame
from geti_sdk.data_models.enums import ShapeType
from geti_sdk.rest_converters import AnnotationRESTConverter
from tests.helpers import SdkTestMode
from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestAnnotationClient:
    sorted_frame_indices_video_1_light_bulbs_project = [
        0,
        29,
        58,
        87,
        116,
        145,
        174,
        203,
    ]
    sorted_frame_indices_video_2_light_bulbs_project = [
        0,
        29,
        58,
        87,
        116,
        145,
        174,
        203,
        232,
    ]

    @staticmethod
    def ensure_test_project(
        project_service: ProjectService, labels: List[str]
    ) -> Project:
        return project_service.get_or_create_project(
            project_name=f"{PROJECT_PREFIX}_annotation_client",
            project_type="classification",
            labels=[labels],
        )

    @pytest.mark.vcr()
    def test_upload_and_retrieve_annotations_for_video(
        self,
        fxt_project_service: ProjectService,
        fxt_light_bulbs_labels: List[str],
        fxt_video_path_1_light_bulbs: str,
        fxt_light_bulbs_annotation_path: str,
        fxt_test_mode: SdkTestMode,
    ):
        """
        Verifies that uploading and retrieving annotations for a video work
        Steps:
        1. Create classification project
        2. Upload video
        3. Upload annotations for video
        4. Fetch annotations from annotation client
        5. Check that the length of the fetched annotations equal the length of the annotations from dataset
        6. Assert that the fetched annotations have the correct frame indices
        """

        # Create classification project
        self.ensure_test_project(
            project_service=fxt_project_service, labels=fxt_light_bulbs_labels
        )

        # Upload video
        video_client = fxt_project_service.video_client
        video = video_client.upload_video(video=fxt_video_path_1_light_bulbs)

        # Upload annotations for video
        annotation_reader = GetiAnnotationReader(
            base_data_folder=fxt_light_bulbs_annotation_path
        )

        annotation_client = fxt_project_service.annotation_client
        annotation_client.annotation_reader = annotation_reader

        if fxt_test_mode != SdkTestMode.OFFLINE:
            time.sleep(1)

        annotation_client.upload_annotations_for_video(video=video, max_threads=1)

        if fxt_test_mode != SdkTestMode.OFFLINE:
            time.sleep(1)

        #  Fetch annotations from annotation client
        annotation_scenes = annotation_client.get_latest_annotations_for_video(
            video=video
        )

        self.__assert_annotation_scenes_for_videos_light_bulbs_project(
            annotation_scenes=annotation_scenes,
            video=video,
            project_labels=fxt_light_bulbs_labels,
            expected_sorted_frame_indices=TestAnnotationClient.sorted_frame_indices_video_1_light_bulbs_project,
        )

    def __assert_annotation_scenes_for_videos_light_bulbs_project(
        self,
        annotation_scenes: List[AnnotationScene],
        video: Video,
        project_labels: List[str],
        expected_sorted_frame_indices: List[int],
    ):
        """
        Assertions for to check if the given annotation scenes match with video 1's annotation scenes for the light bulbs project
        Steps:
        1. Check length
        2. Check Data
        3. Check frames
        4. Check Shape
        5. Check Labels
        """

        #  Check that the length of the fetched annotations equal the length of the annotations from dataset
        assert len(annotation_scenes) == len(expected_sorted_frame_indices)

        #  Assert that the fetched annotations have the correct frame indices
        sorted_frame_indices = self.__get_sorted_frame_indices_from_annotation_scenes(
            annotation_scenes=annotation_scenes
        )

        assert sorted_frame_indices == expected_sorted_frame_indices

        for annotation_scene in annotation_scenes:
            assert annotation_scene.has_data

        assert (
            annotation_scenes[0]
            .annotations[0]
            .shape.is_full_box(
                image_width=video.media_information.width,
                image_height=video.media_information.height,
            )
        )
        assert annotation_scenes[0].annotations[0].shape.type == ShapeType.RECTANGLE
        assert len(annotation_scenes[0].annotations[0].labels) == 1
        assert annotation_scenes[0].annotations[0].labels[0].name in project_labels

    def __get_sorted_frame_indices_from_annotation_scenes(
        self, annotation_scenes: List[AnnotationScene]
    ) -> List[int]:
        frame_indices = []

        for annotation_scene in annotation_scenes:
            frame_indices.append(annotation_scene.media_identifier.frame_index)
            assert len(annotation_scene.annotations) == 1

        return sorted(frame_indices)

    @pytest.mark.vcr()
    def test_upload_and_retrieve_annotations_for_videos(
        self,
        fxt_project_service: ProjectService,
        fxt_light_bulbs_labels: List[str],
        fxt_video_path_1_light_bulbs: str,
        fxt_video_path_2_light_bulbs: str,
        fxt_light_bulbs_annotation_path: str,
        fxt_test_mode: SdkTestMode,
    ):
        """
        Verifies that uploading and retrieving annotations for multiple video's work
        Steps:
        1. Create classification project
        For both videos:
        2. Upload video
        3. Upload annotations for video
        4. Fetch annotations from annotation client
        5. Check that the length of the fetched annotations equal the length of the annotations from data
        6. Assert that the fetched annotations have the correct frame indices
        """
        #  Create classification project
        self.ensure_test_project(
            project_service=fxt_project_service, labels=fxt_light_bulbs_labels
        )

        #  Upload video
        video_client = fxt_project_service.video_client

        video_1 = video_client.upload_video(video=fxt_video_path_1_light_bulbs)
        video_2 = video_client.upload_video(video=fxt_video_path_2_light_bulbs)

        # Upload annotations for video
        annotation_reader = GetiAnnotationReader(
            base_data_folder=fxt_light_bulbs_annotation_path
        )

        annotation_client = fxt_project_service.annotation_client
        annotation_client.annotation_reader = annotation_reader

        annotation_client.upload_annotations_for_videos(
            videos=[video_1, video_2], max_threads=1
        )

        if fxt_test_mode != SdkTestMode.OFFLINE:
            time.sleep(10)

        #  Fetch annotations from annotation client
        annotation_scenes_for_video_1 = (
            annotation_client.get_latest_annotations_for_video(video=video_1)
        )
        annotation_scenes_for_video_2 = (
            annotation_client.get_latest_annotations_for_video(video=video_2)
        )

        self.__assert_annotation_scenes_for_videos_light_bulbs_project(
            annotation_scenes=annotation_scenes_for_video_1,
            video=video_1,
            project_labels=fxt_light_bulbs_labels,
            expected_sorted_frame_indices=TestAnnotationClient.sorted_frame_indices_video_1_light_bulbs_project,
        )
        self.__assert_annotation_scenes_for_videos_light_bulbs_project(
            annotation_scenes=annotation_scenes_for_video_2,
            video=video_2,
            project_labels=fxt_light_bulbs_labels,
            expected_sorted_frame_indices=TestAnnotationClient.sorted_frame_indices_video_2_light_bulbs_project,
        )

    @pytest.mark.vcr()
    def test_upload_and_retrieve_annotations_for_images(
        self,
        fxt_project_service: ProjectService,
        fxt_light_bulbs_labels: List[str],
        fxt_image_path_1_light_bulbs: str,
        fxt_image_path_2_light_bulbs: str,
        fxt_light_bulbs_annotation_path: str,
    ):
        """
        Verifies that uploading and retrieving annotations for multiple images work
        Steps:
        1. Create classification project
        For both images:
        2. Upload image
        3. Upload annotations for image
        5. Fetch annotations from annotation client
        6. Check the media identifiers for each annotation
        """

        # Create classification project
        self.ensure_test_project(
            project_service=fxt_project_service, labels=fxt_light_bulbs_labels
        )

        image_client = fxt_project_service.image_client

        # Upload image
        image_1 = image_client.upload_image(fxt_image_path_1_light_bulbs)
        image_2 = image_client.upload_image(fxt_image_path_2_light_bulbs)

        # Upload annotations for image
        annotation_reader = GetiAnnotationReader(
            base_data_folder=fxt_light_bulbs_annotation_path
        )

        annotation_client = fxt_project_service.annotation_client
        annotation_client.annotation_reader = annotation_reader

        annotation_client.upload_annotations_for_images(
            images=[image_1, image_2], max_threads=1
        )

        # Fetch annotations from annotation client
        annotation_for_image_1 = annotation_client.get_annotation(media_item=image_1)
        annotation_for_image_2 = annotation_client.get_annotation(media_item=image_2)

        # Check the media identifiers for each annotation
        annotation_image_1_identifier = annotation_for_image_1.media_identifier
        annotation_image_2_identifier = annotation_for_image_2.media_identifier

        assert annotation_image_1_identifier == image_1.identifier
        assert annotation_image_2_identifier == image_2.identifier

    @pytest.mark.vcr()
    def test_download_annotations_for_video(
        self,
        fxt_project_service: ProjectService,
        fxt_temp_directory: str,
        fxt_test_mode: SdkTestMode,
    ):
        """
        Verifies that uploading and retrieving annotations for multiple images work
        Steps:
        1. Create classification project
        2. Get first video for light bulbs project
        3. Get annotation scene at frame 0
        4. Download annotation_scenes to temp directory
        5. Check that the length of the fetched annotations equal the length of the annotations from dataset
        6. Read and Retrieve first annotation from directory
        7. De-identify both first and fetched annotation
        8. Compare annotations
        """

        # Get first video for light bulbs project
        video_client = fxt_project_service.video_client
        video = video_client.get_all_videos()[0]

        # Get annotation scene at frame 0
        annotation_client = fxt_project_service.annotation_client
        video_frame = VideoFrame.from_video(video=video, frame_index=0)
        annotation_scene_frame_0 = annotation_client.get_annotation(
            media_item=video_frame
        )

        # Download annotations to test directory
        temp_dir = fxt_temp_directory
        annotations_temp_dir = os.path.join(temp_dir, "annotations")

        annotation_client.download_annotations_for_video(
            video=video, path_to_folder=temp_dir, max_threads=1
        )
        # Get annotations for test directory
        annotation_reader_from_temp_dir = GetiAnnotationReader(
            base_data_folder=annotations_temp_dir
        )

        file_names = annotation_reader_from_temp_dir.get_data_filenames()

        # Check that the length of the fetched annotations equal the length of the
        # annotations from dataset
        assert len(file_names) == len(
            TestAnnotationClient.sorted_frame_indices_video_1_light_bulbs_project
        )

        # Read and Retrieve first annotation from directory
        with open(
            os.path.join(temp_dir, "annotations", f"{video.name}_frame_0.json"), "r"
        ) as file:
            json_annotation_scene = json.load(file)

        json_annotation_scene["media_identifier"] = video.identifier
        downloaded_annotation_scene = AnnotationRESTConverter.from_dict(
            annotation_scene=json_annotation_scene
        )

        # De-identify both first and fetched annotation
        downloaded_annotation_scene.deidentify()
        de_identified_scene = copy.deepcopy(annotation_scene_frame_0)
        de_identified_scene.deidentify()

        # Compare annotations
        assert downloaded_annotation_scene == de_identified_scene
