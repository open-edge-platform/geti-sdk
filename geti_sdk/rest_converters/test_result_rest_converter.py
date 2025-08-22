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

from typing import Any

from geti_sdk.data_models import TestResult
from geti_sdk.utils import deserialize_dictionary


class TestResultRESTConverter:
    """
    Class that handles conversion of Intel® Geti™ REST output for test results to
    objects, and vice-versa
    """

    @staticmethod
    def from_dict(result_dict: dict[str, Any]) -> TestResult:
        """
        Create a TestResult instance from the input dictionary passed in `result_dict`.

        :param result_dict: Dictionary representing a test result on the Intel® Geti™
            server, as returned by the /tests endpoints
        :return: TestResult instance, holding the result data contained in result_dict
        """
        # Need to convert task type to lower case
        task_type = result_dict["model_info"]["task_type"]
        result_dict["model_info"]["task_type"] = task_type.lower()
        return deserialize_dictionary(result_dict, output_type=TestResult)
