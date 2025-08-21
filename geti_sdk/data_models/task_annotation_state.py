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

import attr

from geti_sdk.data_models.enums import AnnotationState
from geti_sdk.data_models.utils import str_to_enum_converter_by_name_or_value


@attr.define
class TaskAnnotationState:
    """
    Representation of the state of an annotation for a particular task in an
    Intel® Geti™ project.
    """

    task_id: str
    state: str | None = attr.field(
        converter=str_to_enum_converter_by_name_or_value(AnnotationState, allow_none=True),
        default=None,
    )
