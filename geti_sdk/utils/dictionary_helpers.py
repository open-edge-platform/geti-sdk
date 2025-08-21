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

from typing import Any, TypeVar

KeyType = TypeVar("KeyType")


def get_dict_key_from_value(input_dict: dict[KeyType, Any], value) -> KeyType | None:
    """
    Return the key associated with `value` in a dictionary `input_dict`. If the value
    is not found in the dictionary, returns None.

    :param input_dict: Dictionary to search in
    :param value: value to search for
    :return: key associated with value if value is in the input_dict, None otherwise
    """
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    try:
        index = values.index(value)
    except ValueError:
        return None
    return keys[index]
