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

from typing import Any, TypeVar, cast, get_args

from attr import fields, has
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ConfigTypeError, MissingMandatoryValue

OutputTypeVar = TypeVar("OutputTypeVar")


def deserialize_dictionary(input_dictionary: dict[str, Any], output_type: type[OutputTypeVar]) -> OutputTypeVar:
    """
    Deserialize an `input_dictionary` to an object of the type passed in `output_type`.

    :param input_dictionary: Dictionary to deserialize
    :param output_type: Type of the object that the dictionary represents, and to
        which the data will be deserialized
    :return: Object of type `output_type`, holding the data passed in
        `input_dictionary`.
    """

    def prune_dict(data: dict, cls: type[Any]) -> dict:
        """Recursively prune a dictionary to match the structure of an attr class."""
        pruned_data = {}
        for attribute in fields(cls):
            key = attribute.name
            if key in data:
                value = data[key]
                if isinstance(value, dict) and has(attribute.type):
                    pruned_data[key] = prune_dict(value, attribute.type)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    try:
                        item_cls = get_args(attribute.type)[0]
                        pruned_data[key] = [prune_dict(x, item_cls) for x in value]
                    except Exception:
                        pruned_data[key] = value
                else:
                    pruned_data[key] = value
        return pruned_data

    filtered_input_dictionary = prune_dict(input_dictionary, output_type)
    model_dict_config = OmegaConf.create(filtered_input_dictionary)
    schema = OmegaConf.structured(output_type)
    schema_error: DataModelMismatchException | None = None
    try:
        values = OmegaConf.merge(schema, model_dict_config)
        output = cast("output_type", OmegaConf.to_object(values))
    except (ConfigKeyError, MissingMandatoryValue, ConfigTypeError) as error:
        schema_error = DataModelMismatchException(
            input_dictionary=filtered_input_dictionary,
            output_data_model=output_type,
            message=error.args[0],
            error_type=type(error),
        )
        output = None
    if schema_error is not None:
        raise schema_error
    return output


class DataModelMismatchException(BaseException):
    """
    Exception raised when a deserialization event fails, meaning that the
    serialized input data does not match the expected schema for the object to
    construct.

    :param output_data_model: Type of the data model to which the data would be
        deserialized
    :param input_dictionary: Input dictionary which would be deserialized
    :param message: Error message that describes the error that occurred during
        deserialization
    :param error_type: Type of the error which occurred during deserialization
    """

    def __init__(
        self,
        input_dictionary: dict,
        output_data_model: type,
        message: str,
        error_type: type,
    ) -> None:
        self.output_data_model = output_data_model
        self.message = message
        self.input_dictionary = input_dictionary
        self.error_type = error_type

    def __str__(self) -> str:
        """
        Return the string representation of the exception.

        :return: String representation of the exception
        """
        return (
            f"Deserialization of input dictionary to object of type "
            f"'{self.output_data_model.__name__}' failed with error: \n\n"
            f"'{self.error_type.__name__}': {self.message}. "
            f"\n\nThe following input data was received for "
            f"deserialization: \n{self.input_dictionary}"
        )
