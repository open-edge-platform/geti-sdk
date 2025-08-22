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

from __future__ import annotations

import os
from collections import UserList
from typing import Any, Generic, TypeVar

from geti_sdk.data_models.media import Image, MediaItem, Video, VideoFrame
from geti_sdk.utils.serialization_helpers import deserialize_dictionary

MediaTypeVar = TypeVar("MediaTypeVar", Image, Video, VideoFrame)


class MediaList(UserList, Generic[MediaTypeVar]):
    """
    A list containing Intel® Geti™ media entities
    """

    @property
    def ids(self) -> list[str]:
        """
        Return a list of unique database IDs for all media items in the media list.
        """
        return [item.id for item in self.data]

    @property
    def names(self) -> list[str]:
        """
        Return a list of filenames for all media items in the media list.
        """
        return [item.name for item in self.data]

    def get_by_id(self, id_value: str) -> MediaItem:
        """
        Return the item with id `id_value` from the media list.
        """
        for item in self.data:
            if item.id == id_value:
                return item
        raise ValueError(f"Media list {self} does not contain item with ID {id_value}.")

    def get_by_filename(self, filename: str) -> MediaItem:
        """
        Return the item with name `filename` from the media list.
        """
        for item in self.data:
            if item.name == filename:
                return item
        raise ValueError(f"Media list {self} does not contain item with filename {filename}.")

    @property
    def media_type(self) -> type[MediaTypeVar]:
        """
        Return the type of the media contained in this list.
        """
        if self.data:
            return type(self.data[0])
        raise ValueError("Cannot deduce media type from empty MediaList")

    @staticmethod
    def from_rest_list(rest_input: list[dict[str, Any]], media_type: type[MediaTypeVar]) -> MediaList[MediaTypeVar]:
        """
        Create a MediaList instance from a list of media entities obtained from the
        Intel® Geti™ /media endpoints.

        :param rest_input: List of dictionaries representing media entities in Intel® Geti™
        :param media_type: Image or Video, type of the media entities that are to be
            converted.

        :return: MediaList holding the media entities contained in `rest_input`,
            where each entity is of type `media_type`
        """
        return MediaList[MediaTypeVar](
            [deserialize_dictionary(media_dict, output_type=media_type) for media_dict in rest_input]
        )

    @property
    def has_duplicate_filenames(self) -> bool:
        """
        Return True if the media list contains at least two items that have the same
        filename, False otherwise.
        """
        filenames = [os.path.basename(name) for name in self.names]
        return len(set(filenames)) != len(filenames)
