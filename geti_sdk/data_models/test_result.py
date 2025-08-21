# Copyright (C) 2023 Intel Corporation
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

import attr

from geti_sdk.data_models.enums import OptimizationType
from geti_sdk.data_models.utils import (
    str_to_datetime,
    str_to_enum_converter,
    str_to_task_type,
)


@attr.define()
class DatasetInfo:
    """
    Container for dataset information, specific to test datasets
    """

    id: str
    is_deleted: bool
    n_frames: int
    n_images: int
    n_samples: int
    name: str


@attr.define()
class JobInfo:
    """
    Container for job information, specific to model testing jobs
    """

    id: str
    status: str

    @property
    def is_done(self) -> bool:
        """
        Return True if the testing job has finished, False otherwise

        :return: True for a finished job, False otherwise
        """
        return self.status.lower() == "done"


@attr.define()
class ModelInfo:
    """
    Container for information related to the model, specific for model tests
    """

    group_id: str
    id: str
    n_labels: int
    task_type: str = attr.field(converter=str_to_task_type)
    template_id: str
    optimization_type: str = attr.field(converter=str_to_enum_converter(OptimizationType))
    version: int
    precision: list[str] | None = None  # Added in Geti v1.9
    task_id: str | None = None  # Added in Geti v2.0


@attr.define()
class Score:
    """
    Container class holding a score resulting from a model testing job. The metric
    contained can either relate to a single label (`label_id` will be assigned) or
    averaged over the dataset as a whole (`label_id` will be None)

    Score values range from 0 to 1
    """

    name: str
    value: float
    label_id: str | None = None


@attr.define()
class TestResult:
    """
    Representation of the results of a model test job that was run for a specific
    model and dataset in an Intel® Geti™ project
    """

    datasets_info: list[DatasetInfo]
    id: str
    job_info: JobInfo
    model_info: ModelInfo
    name: str
    scores: list[Score]
    creation_time: str | None = attr.field(default=None, converter=str_to_datetime)

    def get_mean_score(self) -> Score:
        """
        Return the mean score computed over the full dataset

        :return: Mean score on the dataset
        """
        if not self.job_info.is_done:
            raise ValueError("Unable to retrieve mean model score, the model testing job is not finished yet.")
        return [score for score in self.scores if score.label_id is None][0]
