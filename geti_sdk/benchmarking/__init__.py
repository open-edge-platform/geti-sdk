# Copyright (C) 2023 Intel Corporation
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

The `benchmarking` package contains the
:py:class:`~geti_sdk.benchmarking.benchmarker.Benchmarker` class, which provides
methods for benchmarking models that are trained and deployed with Geti™.

For example, benchmarking local inference rates for your project can help in selecting
the model architecture to use for your project, or in assessing the performance of the
hardware available for inference.

Module contents
---------------

.. automodule:: geti_sdk.benchmarking.benchmarker
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .benchmarker import Benchmarker

__all__ = ["Benchmarker"]
