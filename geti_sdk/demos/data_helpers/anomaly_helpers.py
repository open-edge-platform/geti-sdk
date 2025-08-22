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
import tarfile

from geti_sdk.demos.constants import DEFAULT_DATA_PATH

from .download_helpers import download_file, set_directory_permissions, validate_hash

DEFAULT_MVTEC_PATH = os.path.join(DEFAULT_DATA_PATH, "mvtec")


def is_ad_dataset(target_folder: str = "data") -> bool:
    """
    Check whether a `target_folder` contains an anomaly detection dataset.

    :param target_folder: Directory to check
    :return: True if the directory contains a valid anomaly detection dataset, False
        otherwise
    """
    content = os.listdir(target_folder)
    expected_directories = ["test", "train"]
    expected_files = ["license.txt", "readme.txt"]
    is_dataset = True
    for dirname in expected_directories + expected_files:
        is_dataset &= dirname in content
    return is_dataset


def get_mvtec_dataset_from_path(dataset_path: str = "data") -> str:
    """
    Download the MVTEC AD 'transistor' dataset if not available.

    Note that the MVTec dataset is released under the following licence:

    License:
        MVTec AD dataset is released under the Creative Commons
        Attribution-NonCommercial-ShareAlike 4.0 International License
        (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

    Reference:
        - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
          The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
          Unsupervised Anomaly Detection; in: International Journal of Computer Vision
          129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.
        - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
          A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
          in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
          9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.

    """
    dataset_name = "transistor"
    os.makedirs(dataset_path, exist_ok=True, mode=0o770)
    transistor_dataset_path = os.path.join(dataset_path, dataset_name)
    if os.path.isdir(transistor_dataset_path) and is_ad_dataset(transistor_dataset_path):
        logging.info(f"MVTEC '{dataset_name}' dataset found at path {transistor_dataset_path}")
        return transistor_dataset_path

    logging.info(
        f"MVTEC '{dataset_name}' dataset was not found at path {dataset_path}. Making "
        f"an attempt to download the data..."
    )
    print(
        "Note that the MVTec Anomaly Detection dataset is released "
        "under the following license:\n"
        "License:\n"
        "  MVTec AD dataset is released under the Creative Commons \n"
        "  Attribution-NonCommercial-ShareAlike 4.0 International License \n"
        "  (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).\n"
        "\n"
        "Reference:\n"
        " - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:\n"
        "   The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for\n"
        "   Unsupervised Anomaly Detection; in: International Journal of Computer Vision\n"
        "   129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.\n"
        " - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —\n"
        "   A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;\n"
        "   in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),\n"
        "   9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982."
    )
    archive_name = f"{dataset_name}.tar.xz"
    url = f"https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/{archive_name}"
    # mydrive.ch uses Let's Encrypt certs, verify_cert is set to True
    download_file(url, target_folder=dataset_path, check_valid_archive=False, verify_cert=True)
    archive_path = os.path.join(dataset_path, archive_name)
    validate_hash(
        file_path=archive_path,
        expected_hash="f9c14ab6c802e69899b529da8b9417b319fa39027a0602ec2beeaf2c3a51e5d527248dfdd3c977a0066fb0ed284c6cdb7236e9cc11d06e927e07072496408be3",
    )

    logging.info(f"Extracting the '{dataset_name}' dataset at path {archive_path}...")
    with tarfile.open(archive_path) as tar_file:
        # B202 - hash is validated before processing, file can be considered as trusted
        tar_file.extractall(dataset_path)  # nosec B202

    if not is_ad_dataset(transistor_dataset_path):
        raise ValueError(
            "The dataset was downloaded and extracted successfully, but the directory "
            f"content did not match the expected content. Please ensure that the "
            f"dataset directory at {transistor_dataset_path} contains the expected "
            f"'{dataset_name}' dataset."
        )

    # Fix permissions on extracted files
    set_directory_permissions(transistor_dataset_path)

    logging.info("Cleaning up...")
    os.remove(archive_path)
    return transistor_dataset_path


def get_mvtec_dataset(dataset_path: str | None = None) -> str:
    """
    Check if the MVTEC 'transistor' dataset is present at the specified path. If not,
    this method will attempt to download the dataset to the path specified.

    If no path is passed, this method will check or create the default path: the
    folder 'data' in the top level of the geti-sdk package.

    :param dataset_path: Path to check against.
    :param verbose: True to print detailed output, False to check silently
    :return: Path to the COCO dataset
    """
    if dataset_path is None:
        dataset_path = DEFAULT_MVTEC_PATH
    return get_mvtec_dataset_from_path(dataset_path)
