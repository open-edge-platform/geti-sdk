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

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from geti_sdk.data_models import Project
from geti_sdk.rest_clients import DatasetClient
from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestImageClient:
    @staticmethod
    def ensure_test_project(project_service: ProjectService, labels: list[str]) -> Project:
        return project_service.get_or_create_project(
            project_name=f"{PROJECT_PREFIX}_image_client",
            project_type="detection",
            labels=[labels],
        )

    @pytest.mark.vcr()
    def test_upload_and_delete_image(
        self,
        fxt_project_service: ProjectService,
        fxt_default_labels: list[str],
        fxt_image_path: str,
    ):
        """
        Verifies that uploading an image works as expected

        Steps:
        1. Create detection project
        2. Upload image from file
        3. Load image with OpenCV directly
        4. Assert that the returned image details (filename, media dimensions) are
            correct
        5. Fetch numpy data for image from server
        6. Check that the two image arrays are approximately equal: There may be
            small difference due to image compression. We test for equality by
            asserting that the RMS error between the two numpy image arrays is less
            than 1
        7. Assert that deleting the image works by checking that the number of images
            in the project reduces by one upon deletion of the image
        """
        self.ensure_test_project(project_service=fxt_project_service, labels=fxt_default_labels)
        image_client = fxt_project_service.image_client
        image = image_client.upload_image(fxt_image_path)
        image_numpy = cv2.imread(fxt_image_path)

        assert image.name == os.path.splitext(os.path.basename(fxt_image_path))[0]
        assert image.media_information.width == image_numpy.shape[1]
        assert image.media_information.height == image_numpy.shape[0]

        fetched_numpy = image.get_data(session=fxt_project_service.session)
        assert fetched_numpy.shape == image_numpy.shape
        difference = fetched_numpy.astype(int) - image_numpy.astype(int)

        assert np.sqrt(np.mean(np.square(difference))) < 1

        n_images = len(image_client.get_all_images())
        result = image_client.delete_images([image])
        assert result
        assert len(image_client.get_all_images()) == n_images - 1

    @pytest.mark.vcr()
    def test_upload_image_folder_and_download(
        self,
        fxt_project_service: ProjectService,
        fxt_default_labels: list[str],
        fxt_image_folder: str,
        request: FixtureRequest,
    ):
        """
        Verifies that uploading a folder of images works as expected

        Steps:
        1. Get project and image client
        2. Get list of all images in project
        3. Upload folder containing several images
        4. Get list of all images in project again
        5. Assert that list of images has increased by the number of images in the
            folder
        6. Download the images to a temporary directory
        7. Assert that all images were downloaded
        """

        self.ensure_test_project(project_service=fxt_project_service, labels=fxt_default_labels)
        image_client = fxt_project_service.image_client

        old_images = image_client.get_all_images()
        n_images = len(old_images)

        # Upload folder
        images = image_client.upload_folder(fxt_image_folder, max_threads=1)
        assert len(images) == len(os.listdir(fxt_image_folder))
        assert len(image_client.get_all_images()) == n_images + len(images)

        target_dir = tempfile.mkdtemp()
        request.addfinalizer(lambda: shutil.rmtree(target_dir))

        # Download all images
        image_client.download_all(target_dir, max_threads=1)

        # Assert that all images are downloaded
        downloaded_filenames = os.listdir(os.path.join(target_dir, "images", "Dataset"))
        assert len(downloaded_filenames) == n_images + len(images)
        for image in images + old_images:
            assert image.name + ".jpg" in downloaded_filenames

        # remove images
        image_client.delete_images(images)
        assert len(image_client.get_all_images()) == n_images

    @pytest.mark.vcr()
    def test_upload_from_list(
        self,
        fxt_project_service: ProjectService,
        fxt_default_labels: list[str],
        fxt_image_folder: str,
        request: FixtureRequest,
    ):
        image_client = fxt_project_service.image_client
        old_images = image_client.get_all_images()
        n_old_images = len(old_images)

        # Upload images from list
        image_base_names = [p.name.split(".")[0] for p in Path.glob(Path(fxt_image_folder), "*")]
        n_to_upload = len(image_base_names) // 2
        assert n_to_upload > 0
        images = image_client.upload_from_list(
            fxt_image_folder,
            image_names=image_base_names,
            max_threads=1,
            extension_included=False,
            n_images=n_to_upload,
            image_names_as_full_paths=False,
        )
        assert len(images) == n_to_upload
        assert len(image_client.get_all_images()) == n_old_images + len(images)

    @pytest.mark.vcr()
    def test_download_specific_dataset(
        self,
        fxt_project_service: ProjectService,
        fxt_default_labels: list[str],
        fxt_image_folder: str,
        fxt_image_folder_light_bulbs: str,
        request: FixtureRequest,
    ):
        """
        Verifies that downloading images from a specific dataset works as expected.

        Steps:
        1. Get project and image client
        2. Create two datasets
        3. Upload images to each dataset. One with images from fxt_image_folder and
        one with images from fxt_image_folder_light_bulbs
        4. Download images from each dataset to respective temporary directories
        5. Assert that all images are downloaded correctly for each dataset
        """

        self.ensure_test_project(project_service=fxt_project_service, labels=fxt_default_labels)

        project = fxt_project_service.project
        dataset_client = DatasetClient(
            session=fxt_project_service.session,
            workspace_id=fxt_project_service.workspace_id,
            project=project,
        )
        image_client = fxt_project_service.image_client

        # Create datasets
        dataset1 = dataset_client.create_dataset(name="dataset1")
        dataset2 = dataset_client.create_dataset(name="dataset2")

        # Upload images to datasets
        images_dataset1 = image_client.upload_folder(fxt_image_folder, dataset=dataset1, max_threads=1)
        images_dataset2 = image_client.upload_folder(fxt_image_folder_light_bulbs, dataset=dataset2, max_threads=1)

        assert len(images_dataset1) == len(os.listdir(fxt_image_folder))
        assert len(images_dataset2) == len(os.listdir(fxt_image_folder_light_bulbs))

        # Create temporary directories for downloads
        target_dir1 = tempfile.mkdtemp()
        target_dir2 = tempfile.mkdtemp()
        request.addfinalizer(lambda: shutil.rmtree(target_dir1))
        request.addfinalizer(lambda: shutil.rmtree(target_dir2))

        image_client.download_all(target_dir1, dataset=dataset1, max_threads=1)
        image_client.download_all(target_dir2, dataset=dataset2, max_threads=1)

        # Verify downloads
        downloaded_filenames1 = os.listdir(os.path.join(target_dir1, "images", dataset1.name))
        downloaded_filenames2 = os.listdir(os.path.join(target_dir2, "images", dataset2.name))

        assert len(downloaded_filenames1) == len(images_dataset1)
        assert len(downloaded_filenames2) == len(images_dataset2)

        for image in images_dataset1:
            assert image.name + ".jpg" in downloaded_filenames1

        for image in images_dataset2:
            assert image.name + ".jpg" in downloaded_filenames2
