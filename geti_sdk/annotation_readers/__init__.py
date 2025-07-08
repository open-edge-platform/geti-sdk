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

The `annotation_readers` package contains the
:py:class:`~geti_sdk.annotation_readers.base_annotation_reader.AnnotationReader`
base class, which provides an interface for implementing custom annotation readers.

Annotation readers server to load annotation files in custom formats and convert them
to Intel® Geti™ format, such that they can be uploaded to an Intel® Geti™ project.

Module contents
---------------

.. automodule:: geti_sdk.annotation_readers.base_annotation_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geti_sdk.annotation_readers.geti_annotation_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geti_sdk.annotation_readers.directory_tree_annotation_reader
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .base_annotation_reader import AnnotationReader
from .directory_tree_annotation_reader import DirectoryTreeAnnotationReader
from .geti_annotation_reader import GetiAnnotationReader

__all__ = [
    "AnnotationReader",
    "GetiAnnotationReader",
    "DirectoryTreeAnnotationReader",
]
