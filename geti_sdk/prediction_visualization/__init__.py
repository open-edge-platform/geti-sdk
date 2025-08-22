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
Introduction
------------

The `prediction_visualization` package provides classes for visualizing models predictions and media annotations.
Aditionally, shortend interface to this package is available through the :py:mod:`~geti_sdk.utils.plot_helpers` module.

The main :py:class:`~geti_sdk.prediction_visualization.visualizer.Visualizer` class is a flexible utility class to work
with Geti-SDK Prediction and Annotation object. You can initialize the Visualizer with the desired settings and
then use it to draw the annotations on the input image.

.. code-block:: python

   from geti_sdk import Visualizer

   visualizer = Visualizer(
      show_labels=True,
      show_confidence=True,
      show_count=False,
   )

   # Obtain a prediction from the Intel Geti platfor server or a local deployment.
   ...

   # Visualize the prediction on the input image.
   result = visualizer.draw(
      numpy_image,
      prediction,
      fill_shapes=True,
      confidence_threshold=0.4,
   )
   visualizer.show_in_notebook(result)

In case the Prediction was generated with a model that supports explainable AI functionality,
the Visualizer can also display the explanation for the prediction.

.. code-block:: python
   image_with_saliency_map = visualizer.explain_label(
      numpy_image,
      prediction,
      label_name="Cat",
      opacity=0.5,
      show_predictions=True,
   )
   visualizer.save_image(image_with_saliency_map, "./explained_prediction.jpg")
   visualizer.show_window(image_with_saliency_map)  # When called in a script

Module contents
---------------

.. automodule:: geti_sdk.prediction_visualization.visualizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: geti_sdk.prediction_visualization.shape_drawer.ShapeDrawer
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .visualizer import Visualizer

__all__ = ["Visualizer"]
