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

import cv2
import numpy as np
from IPython.display import display
from PIL import Image as PILImage

from geti_sdk.data_models import Image


def simulate_low_light_image(image: np.ndarray | Image, reduction_factor: float = 0.5) -> np.ndarray:
    """
    Simulate a reduced intensity and exposure time for an input image.
    It does so by reducing the image brightness and adding simulated shot noise to the
    image.

    :param image: Original image
    :param reduction_factor: Brightness reduction factor. Should be in the
        interval [0, 1]
    :return: Copy of the image, simulated for low lighting conditions (lower
        intensity and exposure time).
    """
    if isinstance(image, np.ndarray):
        new_image = image.copy()
    elif isinstance(image, Image):
        new_image = image.numpy.copy()
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'")

    # Reduce brightness
    new_image = cv2.convertScaleAbs(new_image, alpha=reduction_factor, beta=0)

    # Add some shot noise to simulate reduced exposure time
    PEAK = 255 * reduction_factor
    rng = np.random.default_rng(seed=42)
    return np.clip(rng.poisson(new_image / 255.0 * PEAK) / PEAK * 255, 0, 255).astype("uint8")


def display_image_in_notebook(image: np.ndarray | Image, bgr: bool = True):
    """
    Display an image inline in a Juypter notebook.

    :param image: Image to display
    :param bgr: True if the image has its channels in BGR order, False if it
        is in RGB order. Defaults to True
    """
    if isinstance(image, np.ndarray):
        new_image = image.copy()
    elif isinstance(image, Image):
        new_image = image.numpy.copy()
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'")

    result = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB) if bgr else new_image
    img = PILImage.fromarray(result)
    display(img)
