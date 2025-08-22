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

import attr

from geti_sdk.data_models import Annotation
from geti_sdk.data_models.shapes import Rectangle, Shape


@attr.define
class ROI(Annotation):
    """
    A region of interest for a given image. ROIs are generated for
    intermediate tasks in the pipeline of a project, if those tasks produce local
    labels (for instance a detection or segmentation task).
    """

    shape: Rectangle = attr.field(kw_only=True)
    original_shape: Shape = attr.field(kw_only=True)

    @classmethod
    def from_annotation(cls, annotation: Annotation) -> "ROI":
        """
        Convert an Annotation instance into an ROI.

        :param annotation: Annotation to convert to region of interest
        :return: ROI containing the annotation
        """
        return cls(
            labels=annotation.labels,
            shape=annotation.shape.to_roi(),
            original_shape=annotation.shape,
        )

    def to_absolute_coordinates(self, parent_roi: "ROI") -> "ROI":
        """
        Convert the ROI to an ROI in absolute coordinates, given it's parent ROI.

        :param parent_roi: Parent ROI containing the roi instance
        :return: ROI converted to the coordinate system of the parent ROI
        """
        return ROI(
            labels=self.labels,
            shape=self.shape.to_absolute_coordinates(parent_roi.shape),
            original_shape=self.original_shape.to_absolute_coordinates(parent_roi.shape),
        )
