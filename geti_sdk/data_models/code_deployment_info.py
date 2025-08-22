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

from .enums import DeploymentState
from .model import BaseModel
from .utils import str_to_datetime


@attr.define()
class DeploymentModelIdentifier:
    """
    Container class holding the unique identifiers for a model created in an Intel®
    Geti™ project that will be or has been deployed.

    :var model_id: Unique ID of the model to be deployed
    :var model_group_id: Unique ID of the model group to which the model
        belongs
    """

    model_id: str
    model_group_id: str

    def to_dict(self) -> dict[str, str]:
        """
        Return the dictionary representation of the model identifier.

        :return: Dictionary containing the model identifiers
        """
        return attr.asdict(self)

    @classmethod
    def from_model(cls, model: BaseModel) -> "DeploymentModelIdentifier":
        """
        Get the identifiers needed for model deployment for a given `model`

        :param model: Model to retrieve identifiers for
        """
        return cls(model_id=model.id, model_group_id=model.model_group_id)


@attr.define()
class CodeDeploymentInformation:
    """
    Class containing information pertaining to the deployment of an Intel® Geti™
    project.

    :var id: Unique ID of the code deployment
    :var progress: Progress of the deployment creation process
    :var state: State of the deployment creation process.
    :var models: List of model identifiers for the models involved in the deployment
    :var message: Message providing more details on the state of the deployment
    :var creator_id: String containing the ID of the user who created the deployment
    :var creation_time: Time at which the deployment was created
    """

    id: str
    progress: float
    state: DeploymentState
    models: list[DeploymentModelIdentifier]
    creator_id: str
    creation_time: str | None = attr.field(default=None, converter=str_to_datetime)
    message: str = ""
