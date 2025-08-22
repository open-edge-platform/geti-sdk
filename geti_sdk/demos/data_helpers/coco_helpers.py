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
import zipfile
from enum import Enum

from geti_sdk.demos.constants import DEFAULT_DATA_PATH

from .download_helpers import download_file, validate_hash

DEFAULT_COCO_PATH = os.path.join(DEFAULT_DATA_PATH, "coco")


class COCOSubset(Enum):
    """
    Enum representing a certain subset of the MS COCO dataset.
    """

    TRAIN2014 = "train2014"
    VAL2014 = "val2014"
    TEST2014 = "test2014"
    TEST2015 = "test2015"
    TRAIN2017 = "train2017"
    VAL2017 = "val2017"
    TEST2017 = "test2017"
    UNLABELED2017 = "unlabeled2017"

    def __str__(self):
        """
        Return string representation of the COCOSubset instance.
        """
        return self.value

    def get_annotations(self) -> str | None:
        """
        Return the name of the annotation folder for the subset, if any.

        :return: String containing the name of the annotation folder for the subset.
            If subset does not have annotations, returns None.
        """
        if self == COCOSubset.VAL2017 or self == COCOSubset.TRAIN2017:
            return "trainval2017"
        if self == COCOSubset.VAL2014 or self == COCOSubset.TRAIN2014:
            return "trainval2014"
        if (
            self == COCOSubset.TEST2017
            or self == COCOSubset.TEST2015
            or self == COCOSubset.TEST2014
            or self == COCOSubset.UNLABELED2017
        ):
            return None
        return None

    def get_hashes(self) -> tuple[str | None, str | None]:
        """
        Return the expected sha256 hashes for the .zip files containing the images
        and the annotations of the dataset

        Note: If an expected hash is not known for a subset, it will be returned as
            `None`

        :return: Tuple holding the hash digests for the archives, structured like:
            (image_zip, annotation_zip).
        """
        if self == COCOSubset.VAL2017:
            return (
                "9ea554bcf9e6f88876b1157ab38247eb7c1c57564c05c7345a06ac479c6e7a3b9c3825150c189d7d3f2e807c95fd0e07fe90161c563591038e697c846ac76007",
                "3f00c90323ee745b37a9ac040d00f170d49695ed9ffc1d8e0fbd4c5e2d8e9c697fd822b2022df552da5f1892dbcaeb68788416a347b05a20035ed0686f0e1f66",
            )
        return None, None


def directory_has_coco_subset(target_folder: str, coco_subset: COCOSubset) -> bool:
    """
    Check if a certain directory contains a particular subset of the coco dataset.

    :param target_folder: Directory to check
    :param coco_subset: Subset to look for
    :return: True if the directory contains the subset, False otherwise
    """
    if not os.path.exists(target_folder):
        return False
    required_dirs = ["images", "annotations"]
    for directory in required_dirs:
        if directory not in os.listdir(target_folder):
            return False
    image_dir = os.path.join(target_folder, "images")
    subset_image_dir = os.path.join(image_dir, str(coco_subset))
    if not os.path.exists(subset_image_dir):
        return False
    if len(os.listdir(subset_image_dir)) == 0:
        return False
    annotations = coco_subset.get_annotations()
    if annotations is not None:
        annotations_dir_content = os.listdir(os.path.join(target_folder, "annotations"))
        if len(annotations_dir_content) == 0:
            return False
        return any(f"instances_{str(coco_subset)}" in filename for filename in annotations_dir_content)
    return True


def get_coco_dataset_from_path(
    target_folder: str = "data",
    coco_subset: COCOSubset | None = None,
    verbose: bool = False,
) -> str:
    """
    Check if a coco dataset exists in the directory at `target_folder`.
    If no such directory exists, this method will attempt to download the COCO
    dataset to the designated folder.

    :param target_folder: Folder to get the COCO dataset from, or to download the
        dataset into
    :param coco_subset: Optional Subset to download. If passed as None, the script
        will check if any of the coco subsets exists in the folder, and will not
        download any data if at least one subset is found.
    :param verbose: True to print verbose output, False for silent mode
    :return: Path to the COCO dataset
    """
    os.makedirs(target_folder, exist_ok=True, mode=0o770)
    found_subset = None

    if coco_subset is None:
        for subset in COCOSubset:
            if subset.get_annotations() is not None and directory_has_coco_subset(target_folder, subset):
                found_subset = subset
                break
        if found_subset is None:
            found_subset = COCOSubset.VAL2017
    else:
        found_subset = coco_subset

    if directory_has_coco_subset(target_folder=target_folder, coco_subset=found_subset):
        if verbose:
            logging.info(f"COCO dataset (subset: {str(found_subset)}) found at path {target_folder}")
        return target_folder
    logging.info(f"COCO dataset was not found at path {target_folder}, making an attempt to download the data.")

    image_url = f"http://images.cocodataset.org/zips/{str(found_subset)}.zip"
    annotations_name = found_subset.get_annotations()
    if annotations_name is not None:
        annotations_url = f"http://images.cocodataset.org/annotations/annotations_{annotations_name}.zip"
    else:
        if verbose:
            logging.info(f"Unable to download annotations for COCO subset {found_subset}. Downloading images only")
        annotations_url = None

    # Download the zip files
    image_zip = download_file(image_url, target_folder, check_valid_archive=True)
    if annotations_url is not None:
        annotations_zip = download_file(annotations_url, target_folder, check_valid_archive=True)
    else:
        annotations_zip = None

    # Create directories for images and annotations
    image_dir = os.path.join(target_folder, "images")
    os.makedirs(image_dir, exist_ok=True, mode=0o770)
    hashes = found_subset.get_hashes()
    zip_to_extraction_mapping = {image_zip: {"directory": image_dir, "expected_hash": hashes[0]}}

    if annotations_zip is not None:
        annotations_dir = os.path.join(target_folder, "annotations")
        os.makedirs(annotations_dir, exist_ok=True, mode=0o770)
        zip_to_extraction_mapping.update({annotations_zip: {"directory": target_folder, "expected_hash": hashes[1]}})
    else:
        annotations_dir = None

    # Extract images and annotations
    for zipfile_path, data_dictionary in zip_to_extraction_mapping.items():
        if verbose:
            logging.info(f"Extracting {zipfile_path}...")
        if data_dictionary["expected_hash"] is not None:
            validate_hash(zipfile_path, data_dictionary["expected_hash"])
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(data_dictionary["directory"])
        os.remove(zipfile_path)

    image_subset_names = os.listdir(image_dir)
    if annotations_dir is not None:
        # Remove unused annotations
        for filename in os.listdir(annotations_dir):
            filepath = os.path.join(annotations_dir, filename)
            if "instances" not in filename:
                os.remove(filepath)
                continue
            annotation_subset = os.path.splitext(filename.split("_")[1])[0]
            if annotation_subset not in image_subset_names:
                os.remove(filepath)

    if verbose:
        logging.info("COCO dataset downloaded and extracted successfully.")
    return target_folder


def get_coco_dataset(dataset_path: str | None = None, verbose: bool = False) -> str:
    """
    Check if the COCO dataset is present at the specified path. If not,
    this method will attempt to download the dataset to the path specified.

    If no path is passed, this method will check or create the default path: the
    folder 'data' in the top level of the geti-sdk package.

    :param dataset_path: Path to check against.
    :param verbose: True to print detailed output, False to check silently
    :return: Path to the COCO dataset
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    return get_coco_dataset_from_path(dataset_path, verbose=verbose)
