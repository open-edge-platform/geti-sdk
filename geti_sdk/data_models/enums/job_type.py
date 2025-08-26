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

from enum import Enum


class JobType(Enum):
    """
    Enum representing the type of a job on the Geti™ cluster.
    """

    UNDEFINED = "undefined"
    TRAIN = "train"
    INFERENCE = "inference"
    RECONSTRUCT_VIDEO = "reconstruct_video"
    EVALUATE = "evaluate"
    OPTIMIZATION = "optimization"
    OPTIMIZE_POT = "optimize_pot"
    OPTIMIZE_NNCF = "optimize_nncf"
    TEST = "test"
    PREPARE_IMPORT_TO_NEW_PROJECT = "prepare_import_to_new_project"
    PERFORM_IMPORT_TO_NEW_PROJECT = "perform_import_to_new_project"
    PREPARE_IMPORT_TO_EXISTING_PROJECT = "prepare_import_to_existing_project"
    PERFORM_IMPORT_TO_EXISTING_PROJECT = "perform_import_to_existing_project"
    EXPORT_PROJECT = "export_project"
    IMPORT_PROJECT = "import_project"
    EXPORT_DATASET = "export_dataset"

    def __str__(self) -> str:
        """
        Return the string representation of the JobType instance.
        """
        return self.value
