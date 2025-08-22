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
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

import attr
import cv2
import numpy as np

from geti_sdk.data_models.enums import AnnotationKind, MediaType, ShapeType, TaskType

EnumType = TypeVar("EnumType", bound=Enum)


def deidentify(instance: Any):
    """
    Set all identifier fields of an instance of an attr.define decorated class within the
    GETi REST DataModels to None.

    :param instance: Object to deidentify
    """
    for field in attr.fields(type(instance)):
        if field.name in instance._identifier_fields:
            setattr(instance, field.name, None)


def str_to_enum_converter(
    enum: type[EnumType],
) -> Callable[[str | EnumType], EnumType]:
    """
    Construct a converter function to convert an input value into an instance of the
    Enum subclass passed in `enum`.

    :param enum: type of the Enum to which the converter should convert
    :return: Converter function that takes an input value and attempts to convert it
        into an instance of `enum`
    """

    def _converter(input_value: str | EnumType) -> EnumType:
        """
        Convert an input value to an instance of an Enum.

        :param input_value: Value to convert
        :return: Instance of the Enum
        """
        if isinstance(input_value, str):
            return enum(input_value)
        if isinstance(input_value, enum):
            return input_value
        raise ValueError(f"Invalid argument! Cannot convert value {input_value} to Enum {enum.__name__}")

    return _converter


def str_to_optional_enum_converter(
    enum: type[EnumType],
) -> Callable[[str | EnumType], EnumType]:
    """
    Construct a converter function to convert an input value into an instance of the
    Enum subclass passed in `enum`.

    :param enum: type of the Enum to which the converter should convert
    :return: Converter function that takes an input value and attempts to convert it
        into an instance of `enum`
    """

    def _converter(input_value: str | EnumType | None) -> EnumType | None:
        """
        Convert an input value to an instance of an Enum.

        :param input_value: Value to convert
        :return: Instance of the Enum
        """
        if isinstance(input_value, str):
            return enum(input_value)
        if isinstance(input_value, enum):
            return input_value
        if input_value is None:
            return None
        raise ValueError(f"Invalid argument! Cannot convert value {input_value} to Enum {enum.__name__}")

    return _converter


def str_to_enum_converter_by_name_or_value(
    enum: type[EnumType], allow_none: bool = False
) -> Callable[[str | EnumType], EnumType]:
    """
    Construct a converter function to convert an input value into an instance of the
    Enum subclass passed in `enum`.

    This method attempts to convert both from the Enum value as well as it's name

    :param enum: type of the Enum to which the converter should convert
    :param allow_none: True to allow `None` as input value, i.e. for optional parameters
    :return: Converter function that takes an input value and attempts to convert it
        into an instance of `enum`
    """

    def _converter(input_value: str | EnumType) -> EnumType:
        """
        Convert an input value to an instance of an Enum.

        :param input_value: Value to convert
        :return: Instance of the Enum
        """
        if isinstance(input_value, str):
            try:
                return enum(input_value)
            except ValueError:
                return enum[input_value]
        elif isinstance(input_value, enum):
            return input_value
        if (input_value is None) and allow_none:
            return None
        raise ValueError(f"Invalid argument! Cannot convert value {input_value} to Enum {enum.__name__}")

    return _converter


def str_to_task_type(task_type: str | TaskType) -> TaskType:
    """
    Convert an input string to a task type.

    :param task_type:
    :return: TaskType instance corresponding to `task_type`
    """
    if isinstance(task_type, str):
        return TaskType(task_type)
    return task_type


def str_to_media_type(media_type: str | MediaType) -> MediaType:
    """
    Convert an input string to a media type.

    :param media_type:
    :return: MediaType instance corresponding to `media_type`
    """
    if isinstance(media_type, str):
        return MediaType(media_type)
    return media_type


def str_to_shape_type(shape_type: str | ShapeType) -> ShapeType:
    """
    Convert an input string to a shape type.

    :param shape_type:
    :return: ShapeType instance corresponding to `shape_type`
    """
    if isinstance(shape_type, str):
        return ShapeType(shape_type)
    return shape_type


def str_to_annotation_kind(
    annotation_kind: str | AnnotationKind,
) -> AnnotationKind:
    """
    Convert an input string to an annotation kind.

    :param annotation_kind:
    :return: AnnotationKind instance corresponding to `annotation_kind`
    """
    if isinstance(annotation_kind, str):
        return AnnotationKind(annotation_kind)
    return annotation_kind


def str_to_datetime(datetime_str: str | datetime | None) -> datetime | None:
    """
    Convert a string to a datetime.

    :param datetime_str: string containing the datetime, in isoformat.
    :return: datetime instance
    """
    if isinstance(datetime_str, str):
        try:
            if datetime_str.isdigit():
                # POSIX timestamp
                return datetime.fromtimestamp(int(datetime_str) / 1000)
            # ISO format
            return datetime.fromisoformat(datetime_str)
        except ValueError:
            logging.debug(f"Unable to convert str '{datetime_str}' to datetime, converter returns None instead.")
            return None
    elif isinstance(datetime_str, datetime):
        return datetime_str
    elif datetime_str is None:
        return None
    return None


def attr_value_serializer(instance, field, value):
    """
    Convert a value in an attr.define decorated class to string representation, used
    while converting the attrs object to a dictionary.

    Converts Enums and datetime objects to string representation

    :param instance:
    :param field:
    :param value:
    :return:
    """
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, datetime):
        return datetime.isoformat(value)
    return value


def numpy_from_buffer(buffer: bytes) -> np.ndarray:
    """
    Convert a bytes string representing an image into a numpy array.

    :param buffer: Bytes object to convert
    :return: Numpy.ndarray containing the numpy data from the image
    """
    numpy_array = np.frombuffer(buffer, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)


def round_dictionary(input_data: dict[str, Any] | list[Any], decimal_places: int = 3) -> dict[str, Any] | list[Any]:
    """
    Convert all floats in a dictionary to string representation, rounded to
    `decimal_places`.

    :param input_data: Input dictionary to convert and round
    :param decimal_places: Number of decimal places to round to. Defaults to 3
    :return: dictionary with all floats rounded
    """
    if isinstance(input_data, dict):
        for key, value in input_data.items():
            if isinstance(value, float):
                input_data[key] = f"{value:.{decimal_places}f}"
            elif isinstance(value, dict | list):
                input_data[key] = round_dictionary(value, decimal_places=decimal_places)
        return input_data
    if isinstance(input_data, list):
        new_list = []
        for item in input_data:
            if isinstance(item, float):
                new_list.append(f"{item:.{decimal_places}f}")
            elif isinstance(item, dict | list):
                new_list.append(round_dictionary(item, decimal_places=decimal_places))
        return new_list
    return None


def round_to_n_digits(n: int) -> Callable[[float], float]:
    """
    Return a function to round an input number to n digits.

    :param n: Number of digits to round to
    :return: Callable that, when called with a number as input, will round the number
        to n digits.
    """

    def _n_digit_rounder(value: float) -> float:
        return round(value, ndigits=n)

    return _n_digit_rounder


def remove_null_fields(input: Any):  # noqa: A002
    """
    Remove fields that have 'None' or an emtpy string '' as their value from a
    dictionary.

    NOTE: This function modifies the input dictionary in place

    :param input: Dictionary to remove the null fields from
    """
    if isinstance(input, dict):
        for key, value in list(input.items()):
            if isinstance(value, dict):
                remove_null_fields(value)
            elif isinstance(value, Sequence) and not isinstance(value, str):
                for item in value:
                    remove_null_fields(item)
            elif value is None or (isinstance(value, str) and value == ""):
                input.pop(key)
    elif isinstance(input, list):
        for item in input:
            remove_null_fields(item)
