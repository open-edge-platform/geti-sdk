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

from typing import Any, ClassVar

import attr

from geti_sdk.data_models import Algorithm
from geti_sdk.data_models.enums import ConfigurationEntityType
from geti_sdk.data_models.utils import attr_value_serializer, str_to_enum_converter


@attr.define
class EntityIdentifier:
    """
    Identifying information for a configurable entity on the Intel® Geti™ platform,
    as returned by the /configuration endpoint.

    :var workspace_id: ID of the workspace to which the entity belongs
    :var type: Type of the configuration
    """

    _identifier_fields: ClassVar[str] = ["workspace_id"]

    type: str = attr.field(converter=str_to_enum_converter(ConfigurationEntityType))
    workspace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the EntityIdentifier.

        :return: Dictionary representing the EntityIdentifier
        """
        return attr.asdict(self, value_serializer=attr_value_serializer, recurse=True)


@attr.define
class HyperParameterGroupIdentifier(EntityIdentifier):
    """
    Identifying information for a HyperParameterGroup on the Intel® Geti™ platform,
    as returned by the /configuration endpoint.

    :var model_storage_id: ID of the model storage to which the hyper parameter group
        belongs
    :var group_name: Name of the hyper parameter group
    """

    _identifier_fields: ClassVar[str] = ["model_storage_id", "workspace_id"]

    group_name: str = attr.field(kw_only=True)
    model_storage_id: str | None = None
    project_id: str | None = None
    algorithm_name: str | None = attr.field(repr=False, default=None)
    model_template_id: str | None = attr.field(repr=False, default=None)

    def resolve_algorithm(self, algorithm: Algorithm):
        """
        Resolve the algorithm name and id of the model template to which the
        HyperParameterGroup applies.

        :param algorithm: Algorithm instance to which the hyper parameters belong
        :return:
        """
        self.algorithm_name = algorithm.name
        self.model_template_id = algorithm.model_manifest_id


@attr.define
class ComponentEntityIdentifier(EntityIdentifier):
    """
    Identifying information for a configurable Component on the Intel® Geti™ platform,
    as returned by the /configuration endpoint.

    :var component: Name of the component
    :var project_id: ID of the project to which the component belongs
    :var task_id: Optional ID of the task to which the component belongs
    """

    _identifier_fields: ClassVar[str] = ["project_id", "task_id", "workspace_id"]

    component: str = attr.field(kw_only=True)
    project_id: str | None = None
    task_id: str | None = None
