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

from geti_sdk.data_models import ProjectStatus
from geti_sdk.utils import deserialize_dictionary


class StatusRESTConverter:
    """
    Class that handles conversion of Intel® Geti™ REST output for status entities to
    objects
    """

    @staticmethod
    def from_dict(project_status_dict: dict[str, Any]) -> ProjectStatus:
        """
        Create a ProjectStatus instance from the input dictionary passed in
        `project_status_dict`.

        :param project_status_dict: Dictionary representing the status of a project on
            the Intel® Geti™ server
        :return: ProjectStatus instance, holding the status data contained in
            project_status_dict
        """
        return deserialize_dictionary(project_status_dict, output_type=ProjectStatus)
