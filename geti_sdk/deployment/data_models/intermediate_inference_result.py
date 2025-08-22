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

from multiprocessing import Queue, Value

import attr
import numpy as np

from geti_sdk.data_models import Annotation, Label, Prediction

from .region_of_interest import ROI


@attr.define(slots=False)
class IntermediateInferenceResult:
    """
    Inference results for intermediate tasks in the pipeline
    """

    prediction: Prediction
    image: np.ndarray
    rois: list[ROI] | None = None

    def __attrs_post_init__(self):
        """
        Initialize private attributes
        """
        self._infer_queue: Queue[ROI] = Queue()
        self._infer_counter = Value("i", 0)

    @property
    def image_width(self) -> int:
        """
        Return the width of the image to which the InferenceResult applies.

        :return: Integer representing the width of the image, in pixels
        """
        return self.image.shape[1]

    @property
    def image_height(self) -> int:
        """
        Return the height of the image to which the InferenceResult applies.

        :return: Integer representing the height of the image, in pixels
        """
        return self.image.shape[0]

    def filter_rois(self, label: Label | None = None) -> list[ROI]:
        """
        Filter the ROIs for the inference results based on an input label.

        :param label: Label to retrieve the ROIs for. If left as None, all the ROIs
            belonging to the inference result are returned
        :return: List of ROIs containing an object with the specified label
        """
        if label is None:
            return self.rois
        return [roi for roi in self.rois if label.name in roi.label_names]

    def generate_views(self, rois: list[ROI] | None = None) -> list[np.ndarray]:
        """
        Generate a list of image views holding the pixel data for the ROIs produced
        by the last local-label task in the pipeline.

        :param rois: Optional list of ROIs to return the views for. If left as None,
            views for all ROIs are returned.
        :return: List of numpy arrays containing the pixel data for the ROI's in the
            list of ROI's associated with this inference result
        """
        if self.rois is None:
            return [self.image]

        if rois is not None:
            rois_to_get = [roi.shape for roi in self.rois if roi in rois]
        else:
            rois_to_get = [roi.shape for roi in self.rois]

        views: list[np.ndarray] = []
        for roi in rois_to_get:
            y0, y1 = int(roi.y), int(roi.y + roi.height)
            x0, x1 = int(roi.x), int(roi.x + roi.width)
            if len(self.image.shape) == 3:
                views.append(self.image[y0:y1, x0:x1, :])
            elif len(self.image.shape) == 2:
                views.append(self.image[y0:y1, x0:x1])
            else:
                raise ValueError(f"Unexpected image shape: {self.image.shape}. Unable to generate image views")
        return views

    def append_annotation(self, annotation: Annotation, roi: ROI):
        """
        Append an Annotation instance to the prediction results, taking into account
        the ROI for which the annotation was predicted.

        This method can be used to add annotations produced by a downstream local task
        to the prediction results

        :param annotation: Annotation to append to the inference results
        :param roi: ROI in which the prediction was made
        """
        absolute_shape = annotation.shape.to_absolute_coordinates(parent_roi=roi.shape)
        self.prediction.append(Annotation(labels=annotation.labels, shape=absolute_shape))

    def extend_annotations(self, annotations: list[Annotation], roi: ROI):
        """
        Extend the list of annotations for the current prediction results, taking
        into account the ROI for which the annotation was predicted.

        This method can be used to add labels produced by a global downstream task to
        the ROI output of it's upstream local task

        :param annotations: List of annotations holding the labels to append
        :param roi: ROI for which the annotations are predicted
        """
        for annotation in annotations:
            annotation.shape = roi.original_shape
        self.prediction.extend(annotations)

    def add_to_infer_queue(self, roi: ROI):
        """
        Add the ROI to the queue of items to infer

        :param roi: ROI for the item to add to the infer queue
        """
        self._infer_queue.put(roi)

    def get_infer_queue(self) -> list[ROI]:
        """
        Return the full infer queue
        """
        rois: list[ROI] = []
        while not self._infer_queue.empty():
            rois.append(self._infer_queue.get())
        return rois

    def clear_infer_queue(self):
        """
        Reset the infer queue
        """
        self._infer_queue = Queue()

    def increment_infer_counter(self):
        """
        Increase the infer counter by one
        """
        with self._infer_counter.get_lock():
            self._infer_counter.value += 1

    def reset_infer_counter(self):
        """
        Reset the infer counter back to zero
        """
        self._infer_counter = Value("i", 0)

    def all_rois_inferred(self) -> bool:
        """
        Return True if all ROIs in the intermediate result have been inferred
        """
        with self._infer_counter.get_lock():
            return self._infer_counter.value == len(self.rois)
