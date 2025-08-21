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

from typing import Any

from geti_sdk.data_models import Job
from geti_sdk.utils import deserialize_dictionary


class JobRESTConverter:
    """
    Class that handles conversion of Intel® Geti™ REST output for jobs to objects,
    and vice-versa
    """

    @staticmethod
    def from_dict(job_dict: dict[str, Any]) -> Job:
        """
        Create a Job instance from the input dictionary passed in `job_dict`.

        :param job_dict: Dictionary representing a job on the Intel® Geti™ server, as
            returned by the /jobs endpoints
        :return: Job instance, holding the job data contained in job_dict
        """
        # There is an inconsistency in the REST API, the `scores` field was changed
        # from array to object. Preprocess the data to account for that
        if "metadata" in job_dict:
            metadata = job_dict["metadata"]
            if "scores" in metadata:
                scores = metadata["scores"]
                if not isinstance(scores, list):
                    metadata["scores"] = [scores]

        return deserialize_dictionary(job_dict, output_type=Job)
