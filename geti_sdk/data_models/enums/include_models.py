# Copyright (C) 2025 Intel Corporation
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


class IncludeModelsType(str, Enum):
    """
    Enum representing the different include models options
    """

    ALL = "all"
    NONE = "none"
    LATEST_ACTIVE = "latest_active"

    def __str__(self) -> str:
        """
        Return the string representation of the IncludeModelsType instance.
        """
        return self.value
