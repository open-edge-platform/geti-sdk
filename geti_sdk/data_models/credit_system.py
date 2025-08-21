# Copyright (C) 2024 Intel Corporation
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
"""
Credit system-related entities
"""

from copy import deepcopy
from pprint import pformat
from typing import Any

import attr

from geti_sdk.data_models.enums import SubscriptionStatus
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    deidentify,
    remove_null_fields,
    str_to_datetime,
    str_to_enum_converter,
)


@attr.define
class CreditBalance:
    """
    Representation of the Credit Balance in Intel Geti
    """

    incoming: int
    available: int
    blocked: int | None = None


@attr.define
class CreditAccount:
    """
    Representation of the Credit Account in Intel Geti
    """

    _identifier_fields = [
        "id",
        "organization_id",
    ]
    id: str
    organization_id: str
    name: str
    balance: CreditBalance
    created: str = attr.field(converter=str_to_datetime)
    updated: str = attr.field(converter=str_to_datetime)
    expires: str | None = attr.field(
        default=None, converter=str_to_datetime
    )  # renewable account doesn't have an expiration timestamp by default
    renewal_day_of_month: int | None = None
    # renewable quota, for the welcoming one-off account it's None
    renewable_amount: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the object.

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string that represents an overview of the model.

        :return:
        """
        deidentified = deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    def deidentify(self) -> None:
        """
        Remove unique database IDs from the base object.
        """
        deidentify(self)


@attr.define
class Subscription:
    """
    Representation of the Subscription in Intel Geti
    """

    _identifier_fields = [
        "id",
        "product_id",
        "organization_id",
        "workspace_id",
    ]
    id: str
    product_id: str
    organization_id: str
    workspace_id: str
    status: SubscriptionStatus = attr.field(converter=str_to_enum_converter(SubscriptionStatus))
    created: str = attr.field(converter=str_to_datetime)
    updated: str = attr.field(converter=str_to_datetime)
    next_renewal_date: str | None = attr.field(default=None, converter=str_to_datetime)
    previous_renewal_date: str | None = attr.field(default=None, converter=str_to_datetime)

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the object.

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string that represents an overview of the model.

        :return:
        """
        deidentified = deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    def deidentify(self) -> None:
        """
        Remove unique database IDs from the base object.
        """
        deidentify(self)
