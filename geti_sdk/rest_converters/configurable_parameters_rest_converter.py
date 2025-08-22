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
from typing import Any

from pydantic import BaseModel

PYDANTIC_BASE_TYPES = {
    "integer",
    "number",
    "boolean",
    "string",
}
PYDANTIC_ANY_OF = "anyOf"


class ConfigurableParametersRESTConverter:
    """
    Base class for converting configurable parameters to REST views.

    This class provides methods to transform Pydantic models and their fields
    into REST-compatible dictionary representations.
    """

    @classmethod
    def configurable_parameters_to_rest(
        cls, configurable_parameters: BaseModel
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Convert a Pydantic model of configurable parameters to its REST representation.

        This method processes a Pydantic model containing configuration parameters and transforms it
        into a REST view. It handles both simple fields and nested models:

        - Simple fields (int, float, str, bool) are converted to a list of dictionaries with metadata
            including key, name, description, value, type, and constraints
        - Nested Pydantic models are processed recursively and maintained as nested structures

        The return format depends on the content:
        - If only simple parameters exist: returns a list of parameter dictionaries
        - If only nested models exist: returns a dictionary mapping nested model names to their contents
        - If both exist: returns a list containing parameter dictionaries and nested model dictionary

        :param configurable_parameters: Pydantic model containing configurable parameters
        :return: REST representation as either a dictionary of nested models,
            a list of parameter dictionaries, or a combined list of both
        """
        nested_params: dict[str, Any] = {}
        list_params: list[dict[str, Any]] = []

        for field_name in configurable_parameters.model_fields:
            if field_name.startswith("allowed_values_"):
                # Skip fields that are not part of the main configurable parameters
                continue

            field = getattr(configurable_parameters, field_name)
            if isinstance(field, BaseModel):
                # If the field is a nested Pydantic model, process it recursively
                nested_params[field_name] = cls.configurable_parameters_to_rest(configurable_parameters=field)
            else:
                # If the field is a simple type, convert directly to REST view
                list_params.append(
                    {
                        "key": field_name,
                        "value": field,
                    }
                )

        # Return combined or individual results based on content
        if nested_params and list_params:
            return [*list_params, nested_params]
        return list_params or nested_params

    @classmethod
    def configurable_parameters_from_rest(
        cls, configurable_parameters_rest: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Convert a REST representation back to a dictionary that can be used to create/update a Pydantic model.

        This method performs the reverse operation of configurable_parameters_to_rest:
        - For a list of parameter dictionaries, it extracts the key-value pairs
        - For a dictionary of nested models, it processes each nested model recursively
        - For a mixed list containing both, it handles both types

        :param configurable_parameters_rest: REST representation as a dictionary or list
        :return: Dictionary representation suitable for Pydantic model instantiation
        """
        # If the input is a list (of parameters or mixed)
        if isinstance(configurable_parameters_rest, list):
            result = {}

            for item in configurable_parameters_rest:
                # If this is a parameter entry (has a "key" field)
                if isinstance(item, dict) and "key" in item:
                    key = item["key"]
                    value = item["value"]
                    result[key] = value
                # If it's a dictionary without a "key" field, it must contain nested models
                elif isinstance(item, dict):
                    # Process each nested model recursively and merge with result
                    nested_result = cls.configurable_parameters_from_rest(item)
                    result.update(nested_result)

            return result

        # If the input is a dictionary (of nested models or other fields)
        if isinstance(configurable_parameters_rest, dict):
            result = {}

            for key, value in configurable_parameters_rest.items():
                # If the value is a complex structure, process it recursively
                if isinstance(value, dict | list):  # Replace `dict | list` with `(dict, list)`
                    result[key] = cls.configurable_parameters_from_rest(value)
                else:
                    # Simple value, keep as is
                    result[key] = value

            return result

        # If it's neither a list nor a dictionary, return it as is
        return configurable_parameters_rest
