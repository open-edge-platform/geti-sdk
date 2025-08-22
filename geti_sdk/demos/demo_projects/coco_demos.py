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

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.demos.data_helpers.coco_helpers import get_coco_dataset
from geti_sdk.rest_clients import ProjectClient
from geti_sdk.utils import get_task_types_by_project_type
from tests.helpers import DatumAnnotationReader

from .utils import ensure_project_is_trained

DEMO_LABELS = ["dog"]
DEMO_PROJECT_TYPE = "detection"
DEMO_PROJECT_NAME = "COCO dog detection"


def create_segmentation_demo_project(
    geti: Geti,
    n_images: int,
    n_annotations: int = -1,
    auto_train: bool = False,
    dataset_path: str | None = None,
    project_name: str = "Segmentation demo",
) -> Project:
    """
    Create a demo project of type 'segmentation', based off the MS COCO dataset.

    This method creates a project with a single 'Segmentation' task, with segmentation
    mask for labels: 'backpack' and 'suitcase'.

    :param geti: Geti instance, representing the GETi server on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the geti_sdk package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :param project_name: Name of the project to create
    :return: Project object, holding detailed information about the project that was
        created on the Intel® Geti™ server.
    """
    coco_path = get_coco_dataset(dataset_path)
    logging.info(" ------- Creating segmentation project --------------- ")

    labels_of_interest = ["backpack", "suitcase"]
    project_type = "segmentation"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(base_data_folder=coco_path, annotation_format="coco")
    annotation_reader.filter_dataset(labels=labels_of_interest, criterion="OR")
    # Create project and upload data
    return geti.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train,
    )


def create_detection_demo_project(
    geti: Geti,
    n_images: int,
    n_annotations: int = -1,
    auto_train: bool = False,
    dataset_path: str | None = None,
    project_name: str = "Detection demo",
) -> Project:
    """
    Create a demo project of type 'detection', based off the MS COCO dataset.

    This method creates a project with a single 'Detection' task, with labels and
    annotations for 'cell phone' and 'person' objects

    :param geti: Geti instance, representing the GETi server on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the geti_sdk package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :param project_name: Name of the project to create
    :return: Project object, holding detailed information about the project that was
        created on the Intel® Geti™ server.
    """
    coco_path = get_coco_dataset(dataset_path)
    logging.info(" ------- Creating detection project --------------- ")

    labels_of_interest = ["cell phone", "person"]
    project_type = "detection"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(base_data_folder=coco_path, annotation_format="coco")
    annotation_reader.filter_dataset(labels=labels_of_interest, criterion="AND")
    # Create project and upload data
    return geti.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train,
    )


def create_classification_demo_project(
    geti: Geti,
    n_images: int,
    n_annotations: int = -1,
    auto_train: bool = False,
    dataset_path: str | None = None,
    project_name: str = "Classification demo",
) -> Project:
    """
    Create a demo project of type 'classification', based off the MS COCO dataset.

    This method creates a project with a single 'Classification' task, with labels to
    classify images with different types of animals. The following animals are
    considered: "horse", "cat", "zebra", "bear"

    :param geti: Geti instance, representing the GETi server on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the geti_sdk package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :param project_name: Name of the project to create
    :return: Project object, holding detailed information about the project that was
        created on the Intel® Geti™ server.
    """
    coco_path = get_coco_dataset(dataset_path)
    logging.info(" ------- Creating classification project --------------- ")

    labels_of_interest = ["horse", "cat", "zebra", "bear"]
    project_type = "classification"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(base_data_folder=coco_path, annotation_format="coco")
    annotation_reader.filter_dataset(labels=labels_of_interest, criterion="XOR")
    # Create project and upload data
    return geti.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train,
    )


def create_detection_to_segmentation_demo_project(
    geti: Geti,
    n_images: int,
    n_annotations: int = -1,
    auto_train: bool = False,
    dataset_path: str | None = None,
    project_name: str = "Animal detection to segmentation demo",
) -> Project:
    """
    Create a demo project of type 'detection_to_segmentation', based off the MS COCO
    dataset.

    This method creates a project with a 'Detection' task, followed by a 'Segmentation'
    task. The detection task has the label 'animal', and the segmentation task has
    annotations for the following species:
        "dog", "cat", "horse", "elephant", "cow", "sheep", "giraffe", "zebra", "bear"


    :param geti: Geti instance, representing the GETi server on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the geti_sdk package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :param project_name: Name of the project to create
    :return: Project object, holding detailed information about the project that was
        created on the Intel® Geti™ server.
    """
    coco_path = get_coco_dataset(dataset_path)
    logging.info(" ------- Creating detection -> segmentation project --------------- ")
    animal_labels = ["dog", "cat", "horse", "cow", "sheep"]
    project_type = "detection_to_segmentation"

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(project_type):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=coco_path, annotation_format="coco", task_type=task_type
        )
        annotation_reader.filter_dataset(labels=animal_labels, criterion="OR")
        label_source_per_task.append(annotation_reader)

    # Group the labels for the first annotation reader, so that the detection task
    # will only see a single label
    label_source_per_task[0].group_labels(labels_to_group=animal_labels, group_name="animal")

    # Create project and upload data
    return geti.create_task_chain_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train,
    )


def create_detection_to_classification_demo_project(
    geti: Geti,
    n_images: int,
    n_annotations: int = -1,
    auto_train: bool = False,
    dataset_path: str | None = None,
    project_name: str = "Animal detection to classification demo",
) -> Project:
    """
    Create a demo project of type 'detection_to_classification', based off the MS COCO
    dataset.

    This method creates a project with a 'Detection' task, followed by a
    'Classification' task. The detection task has the label 'animal', and the
    classification task discriminates between 'domestic' and 'wild' animals.

    :param geti: Geti instance, representing the GETi server on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the geti_sdk package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :param project_name: Name of the project to create
    :return: Project object, holding detailed information about the project that was
        created on the Intel® Geti™ server.
    """
    coco_path = get_coco_dataset(dataset_path)
    logging.info(" ------- Creating detection -> classification project --------------- ")
    domestic_labels = ["dog", "cat", "horse", "cow", "sheep"]
    wild_labels = ["elephant", "giraffe", "zebra", "bear"]
    animal_labels = domestic_labels + wild_labels

    project_type = "detection_to_classification"

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(project_type):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=coco_path, annotation_format="coco", task_type=task_type
        )
        annotation_reader.filter_dataset(labels=animal_labels, criterion="OR")
        label_source_per_task.append(annotation_reader)

    # Group the labels for the first annotation reader, so that the detection task
    # will only see a single label
    label_source_per_task[0].group_labels(labels_to_group=animal_labels, group_name="animal")
    # Group the labels for the second annotation reader
    label_source_per_task[1].group_labels(labels_to_group=domestic_labels, group_name="domestic")
    label_source_per_task[1].group_labels(labels_to_group=wild_labels, group_name="wild")
    # Create project and upload data
    return geti.create_task_chain_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train,
    )


def ensure_trained_example_project(geti: Geti, project_name: str = DEMO_PROJECT_NAME) -> Project:
    """
    Ensure that the project specified by `project_name` exists on the GETi
    instance addressed by `geti`.

    :param geti: Geti instance pointing to the GETi server
    :param project_name: Name of the project
    :return: Project object, representing the project in GETi
    """
    project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)
    project = project_client.get_project_by_name(project_name=project_name)

    if project is None:
        # There are two options: Either the user has used the default project name
        # but has not run the `create_coco_project_single_task.py` example yet, or the
        # user has specified a different project which simply doesn't exist.
        #
        # In the first case, we create the project
        #
        # In the second case, we raise an error stating that the project doesn't exist
        # and should be created first
        if project_name == DEMO_PROJECT_NAME:
            print(f"\nThe project `{project_name}` does not exist on the server yet, creating it now.... \n")
            coco_path = get_coco_dataset()

            # Create annotation reader
            annotation_reader = DatumAnnotationReader(base_data_folder=coco_path, annotation_format="coco")
            annotation_reader.filter_dataset(labels=DEMO_LABELS, criterion="OR")
            # Create project and upload data
            project = geti.create_single_task_project_from_dataset(
                project_name=project_name,
                project_type=DEMO_PROJECT_TYPE,
                path_to_images=coco_path,
                annotation_reader=annotation_reader,
                labels=DEMO_LABELS,
                number_of_images_to_upload=50,
                number_of_images_to_annotate=45,
                enable_auto_train=True,
            )
            # Should wait for some time for the job to appear as scheduled before checking if
            # the project is trained. Auto training is triggered after around 5 seconds.
            print("Project created. Waiting for training job to be scheduled. This may take a few seconds.")
            time.sleep(5)
        else:
            raise ValueError(
                f"The project named `{project_name}` does not exist on the server at "
                f"`{geti.session.config.host}`. Please either create it first, or "
                f"specify an existing project."
            )

    ensure_project_is_trained(geti, project)
    return project
