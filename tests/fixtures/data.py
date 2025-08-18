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
import json
import os
import shutil
from typing import Callable, List

import pytest

from geti_sdk.annotation_readers import GetiAnnotationReader
from tests.helpers import DatumAnnotationReader


@pytest.fixture(scope="session")
def fxt_blocks_dataset(fxt_base_test_path) -> str:
    """
    This fixture returns the path to the 'blocks' dataset
    """
    yield os.path.join(fxt_base_test_path, "data", "blocks")


@pytest.fixture(scope="session")
def fxt_light_bulbs_dataset(fxt_base_test_path) -> str:
    """
    This fixture returns the path to the 'light-bulbs' dataset
    """
    yield os.path.join(fxt_base_test_path, "data", "../data/light-bulbs")


@pytest.fixture(scope="session")
def fxt_human_pose_dataset(fxt_base_test_path) -> str:
    """
    This fixture returns the path to the 'human-pose' dataset
    """
    yield os.path.join(fxt_base_test_path, "data", "human-pose")


@pytest.fixture(scope="session")
def fxt_human_pose_image_folder(fxt_human_pose_dataset) -> str:
    """
    This fixture returns the path to the 'human-pose' dataset
    """
    yield os.path.join(fxt_human_pose_dataset, "images", "default")


@pytest.fixture(scope="session")
def fxt_image_folder(fxt_blocks_dataset) -> str:
    """
    This fixture returns the path to a sample image
    """
    yield os.path.join(fxt_blocks_dataset, "images", "NONE")


@pytest.fixture(scope="session")
def fxt_video_folder_light_bulbs(fxt_light_bulbs_dataset) -> str:
    """
    This fixture returns the path to the videos folder in the light bulbs dataset
    """
    yield os.path.join(fxt_light_bulbs_dataset, "videos")


@pytest.fixture(scope="session")
def fxt_image_folder_light_bulbs(fxt_light_bulbs_dataset) -> str:
    """
    This fixture returns the path to the images folder in the light bulbs dataset
    """
    yield os.path.join(fxt_light_bulbs_dataset, "images")


@pytest.fixture(scope="session")
def fxt_image_path(fxt_image_folder) -> str:
    """
    This fixture returns the path to a sample image
    """
    yield os.path.join(fxt_image_folder, "WIN_20220406_21_24_24_Pro.jpg")


@pytest.fixture(scope="session")
def fxt_video_path_1_light_bulbs(fxt_video_folder_light_bulbs) -> str:
    """
    This fixture returns the path to a sample video from the light bulbs dataset
    """
    yield os.path.join(fxt_video_folder_light_bulbs, "Light-Bulb1.mp4")


@pytest.fixture(scope="session")
def fxt_video_path_2_light_bulbs(fxt_video_folder_light_bulbs) -> str:
    """
    This fixture returns the path to a sample video from the light bulbs dataset
    """
    yield os.path.join(fxt_video_folder_light_bulbs, "Light-Bulb2.mp4")


@pytest.fixture(scope="session")
def fxt_image_path_1_light_bulbs(fxt_image_folder_light_bulbs) -> str:
    """
    This fixture returns the path to a sample image from the light bulbs dataset
    """
    yield os.path.join(fxt_image_folder_light_bulbs, "lamp-1523123.jpg")


@pytest.fixture(scope="session")
def fxt_image_path_2_light_bulbs(fxt_image_folder_light_bulbs) -> str:
    """
    This fixture returns the path to a sample image from the light bulbs dataset
    """
    yield os.path.join(fxt_image_folder_light_bulbs, "light-bulb-on-yellow-1426164.jpg")


@pytest.fixture(scope="session")
def fxt_light_bulbs_annotation_path(fxt_light_bulbs_dataset) -> str:
    """
    This fixture returns the path to annotations from the light bulbs project
    """
    yield os.path.join(fxt_light_bulbs_dataset, "annotations")


@pytest.fixture(scope="session")
def fxt_image_path_complex(fxt_image_folder) -> str:
    """
    This fixture returns the path to a complex sample image
    """
    yield os.path.join(fxt_image_folder, "WIN_20220406_21_26_02_Pro.jpg")


@pytest.fixture(scope="function")
def fxt_annotation_reader(fxt_blocks_dataset) -> DatumAnnotationReader:
    """
    This fixture returns a Datumaro Annotation Reader which can read annotations for
    the 'blocks' dataset
    """
    yield DatumAnnotationReader(
        base_data_folder=fxt_blocks_dataset, annotation_format="coco"
    )


@pytest.fixture(scope="function")
def fxt_annotation_reader_grouped(fxt_blocks_dataset) -> DatumAnnotationReader:
    """
    This fixture returns a Datumaro Annotation Reader which can read annotations for
    the 'blocks' dataset. All labels in this reader have been grouped to a single
    'blocks' label.
    """
    reader = DatumAnnotationReader(
        base_data_folder=fxt_blocks_dataset, annotation_format="coco"
    )
    reader.group_labels(labels_to_group=["cube", "cylinder"], group_name="block")
    yield reader


@pytest.fixture(scope="function")
def fxt_annotation_reader_keypoint(fxt_human_pose_dataset) -> DatumAnnotationReader:
    """
    This fixture returns a Datumaro Annotation Reader which can read annotations for
    the 'keypoint' dataset
    """
    yield DatumAnnotationReader(
        base_data_folder=fxt_human_pose_dataset, annotation_format="datumaro"
    )


@pytest.fixture(scope="function")
def fxt_annotation_reader_grouped_keypoint(
    fxt_human_pose_dataset,
) -> DatumAnnotationReader:
    """
    This fixture returns a Datumaro Annotation Reader which can read annotations for
    the 'keypoint' dataset. All labels in this reader have been grouped to a single
    'blocks' label.
    """
    reader = DatumAnnotationReader(
        base_data_folder=fxt_human_pose_dataset, annotation_format="datumaro"
    )
    reader.group_labels(labels_to_group=["cube", "cylinder"], group_name="block")
    yield reader


@pytest.fixture(scope="function")
def fxt_annotation_reader_factory(
    fxt_blocks_dataset,
) -> Callable[[None], DatumAnnotationReader]:
    """
    This fixutre returns Datumaro Annotation Readers which can read annotations for
    the 'blocks' dataset. The fixture can be called multiple times to yield different
    instances of the annotation reader
    """

    def _create_annotation_reader() -> DatumAnnotationReader:
        return DatumAnnotationReader(
            base_data_folder=fxt_blocks_dataset, annotation_format="coco"
        )

    yield _create_annotation_reader


@pytest.fixture(scope="function")
def fxt_project_dictionary(fxt_base_test_path: str) -> dict:
    """
    Return a dictionary representing a project.
    """
    project_datafile = os.path.join(fxt_base_test_path, "data", "mock_project.json")
    with open(project_datafile, "r") as file:
        project_dict = json.load(file)
    yield project_dict


@pytest.fixture(scope="function")
def fxt_env_filepath(fxt_base_test_path: str) -> str:
    """
    Return the path to a mock .env file
    """
    yield os.path.join(fxt_base_test_path, "data", "mock.env")


@pytest.fixture()
def fxt_geti_annotation_reader(fxt_light_bulbs_dataset) -> GetiAnnotationReader:
    """
    Return a GetiAnnotationReader instance to load the `light-bulbs` test dataset
    """
    yield GetiAnnotationReader(
        base_data_folder=os.path.join(fxt_light_bulbs_dataset, "annotations")
    )


@pytest.fixture()
def fxt_classification_to_detection_annotation_readers(
    fxt_annotation_reader: DatumAnnotationReader,
    fxt_annotation_reader_grouped: DatumAnnotationReader,
) -> List[DatumAnnotationReader]:
    """
    Return a list of DatumAnnotationReader objects that can be used to create a
    `classification_to_detection` project, for the `blocks` dataset.
    """
    yield [fxt_annotation_reader_grouped, fxt_annotation_reader]


@pytest.fixture(scope="session")
def fxt_dice_dataset(fxt_base_test_path) -> str:
    """
    This fixture returns the path to the 'dice' dataset for video prediction
    """
    yield os.path.join(fxt_base_test_path, "data", "dice")


@pytest.fixture(scope="session")
def fxt_deployment_path_dice(fxt_dice_dataset) -> str:
    """
    This fixture returns the path to the deployment folder for video prediction
    """
    deployment_path = os.path.join(fxt_dice_dataset, "deployment")
    deployment_file = os.path.join(fxt_dice_dataset, "deployment.zip")
    shutil.unpack_archive(deployment_file, deployment_path)
    yield deployment_path
    shutil.rmtree(deployment_path)


@pytest.fixture(scope="session")
def fxt_video_path_dice(fxt_dice_dataset) -> str:
    """
    This fixture returns the path to a sample video for video prediction
    """
    yield os.path.join(fxt_dice_dataset, "dice_rolling_with_sound.mp4")
