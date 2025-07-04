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

"""
Introduction
------------

The `utils` package contains utility functions used throughout the SDK.

Module contents
---------------
"""

from .algorithm_helpers import get_default_algorithm_info, get_supported_algorithms
from .credentials_helpers import get_server_details_from_env
from .deprecation import deprecate
from .dictionary_helpers import get_dict_key_from_value
from .label_helpers import (
    generate_classification_labels,
    generate_segmentation_labels,
    generate_unique_label_color,
)
from .plot_helpers import (
    show_image_with_annotation_scene,
    show_video_frames_with_annotation_scenes,
)
from .project_helpers import get_project_folder_name, get_task_types_by_project_type
from .serialization_helpers import deserialize_dictionary
from .workspace_helpers import get_workspace_id

__all__ = [
    "deprecate",
    "get_workspace_id",
    "generate_classification_labels",
    "generate_segmentation_labels",
    "get_dict_key_from_value",
    "get_task_types_by_project_type",
    "show_image_with_annotation_scene",
    "show_video_frames_with_annotation_scenes",
    "get_supported_algorithms",
    "deserialize_dictionary",
    "get_server_details_from_env",
    "get_project_folder_name",
    "generate_unique_label_color",
    "get_default_algorithm_info",
]
