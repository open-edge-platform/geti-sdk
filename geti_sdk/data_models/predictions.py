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

from datetime import datetime
from typing import Any, ClassVar

import attr
import numpy as np

from geti_sdk.data_models.annotation_scene import AnnotationScene
from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.enums import AnnotationKind
from geti_sdk.data_models.label import Label
from geti_sdk.data_models.media import MediaInformation
from geti_sdk.data_models.utils import (
    deidentify,
    str_to_annotation_kind,
    str_to_datetime,
)
from geti_sdk.http_session import GetiSession


@attr.define
class ResultMedium:
    """
    Representation of a single result medium in Intel® Geti™.

    :var name: Name of the result medium option
    :var type: Type of the result medium represented by this object
    :var url: URL at which the full result medium can be downloaded
    :var id: Unique database ID assigned to the result medium
    :var label_id: Unique database ID of the label referenced by this result medium
    """

    _identifier_fields: ClassVar[str] = ["id", "data", "label_id", "url"]

    name: str
    type: str | None = None
    url: str | None = None
    label_id: str | None = None
    id: str | None = None
    roi: dict[str, Any] | None = None
    label_name: str | None = attr.field(init=False, default=None)
    data: bytes | None = attr.field(default=None, repr=False, init=False)

    def resolve_label_name(self, labels: list[Label]):
        """
        Add the label name to the result medium, by matching the label_id to a list of
        Labels.

        :param labels: List of Labels to get the name from
        """
        self.label_name = next((label.name for label in labels if label.id == self.label_id), None)
        # Label id is not defined for anomaly classification, it will always return
        # the saliency map for the 'Anomalous' label
        if self.name.lower() == "anomaly map" and self.label_name is None:
            self.label_name = "Anomalous"

    def get_data(self, session: GetiSession) -> bytes:
        """
        Download the data belonging to this ResultMedium object.

        :param session: REST session to the Intel® Geti™ server from which this ResultMedium
            was generated
        :return: bytes object holding the data, if any is found
        """
        if self.data is None and self.url is not None:
            response = session.get_rest_response(url=self.url, method="GET", contenttype="jpeg")
            if response.status_code == 200:
                self.data = response.content
            else:
                raise ValueError(
                    f"Unable to retrieve data for result medium {self}, received "
                    f"response {response} from Intel® Geti™ server."
                )
        return self.data

    @property
    def friendly_name(self) -> str:
        """
        Return a human readable name with which the result medium can be identified.

        :return: friendly name for the result medium
        """
        return self.name + "_" + self.label_name if self.label_name is not None else self.name


@attr.define
class Prediction(AnnotationScene):
    """
    Representation of the model predictions for a certain media entity in Intel® Geti™.

    :var annotations: List of predictions belonging to the media entity
    :var id: unique database ID of the Prediction in Intel® Geti™
    :var kind: Kind of prediction (Annotation or Prediction)
    :var media_identifier: Identifier of the media entity to which this Prediction
        applies
    :var modified: Date and time at which this Prediction was last modified
    :var maps: List of additional result media belonging to this prediction
    :var feature_vector: Optional feature vector (produced by the model) for the image
        to which the prediction relates
    :var active_score: Optional active score (produced by the model) for the image
        to which the prediction relates
    """

    kind: str = attr.field(
        converter=str_to_annotation_kind,
        default=AnnotationKind.PREDICTION,
        kw_only=True,
    )
    maps: list[ResultMedium] = attr.field(factory=list, kw_only=True)
    feature_vector: np.ndarray | None = attr.field(kw_only=True, default=None, repr=False)
    created: str | None = attr.field(converter=str_to_datetime, default=None)

    def resolve_labels_for_result_media(self, labels: list[Label]) -> None:
        """
        Resolve the label names for all result media available with this Prediction.

        :param labels: List of Labels for the project, from which the names are taken
        """
        for map_ in self.maps:
            map_.resolve_label_name(labels=labels)

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the prediction and the entities it
        contains.
        """
        deidentify(self)
        self.media_identifier = None
        for annotation in self.annotations:
            annotation.deidentify()
        for map_ in self.maps:
            deidentify(map_)

    @property
    def has_result_media(self) -> bool:
        """
        Return True if this Prediction has result media belonging to it, False
        otherwise.

        :return: True if there are result media belonging to the prediction
        """
        return len(self.maps) > 0

    def get_result_media_data(self, session: GetiSession) -> list[ResultMedium]:
        """
        Download the data for all result media belonging to this prediction.

        :param session: REST session to the Intel® Geti™ server from which this Prediction
            was generated
        :return: List of result media, that have their data downloaded from the cluster
        """
        result: list[ResultMedium] = []
        for medium in self.maps:
            medium.get_data(session=session)
            result.append(medium)
        return result

    def as_mask(
        self,
        media_information: MediaInformation,
        probability_threshold: float | None = None,
    ) -> np.ndarray:
        """
        Convert the shapes in the prediction to a mask that can be overlayed on an
        image.

        :param media_information: MediaInformation object containing the width and
            heigth of the image for which the mask should be generated.
        :param probability_threshold: Threshold (between 0 and 1) for the probability.
            Shapes that are predicted with a probability below this threshold will
            not be plotted in the mask. If left as None (the default), all predictions
            will be shown.
        :return: np.ndarray holding the mask representation of the prediction
        """
        image_width = media_information.width
        image_height = media_information.height
        mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        for annotation in self.annotations:
            max_prob_label_index = int(np.argmax([label.probability for label in annotation.labels]))
            max_prob_label = annotation.labels[max_prob_label_index]
            if probability_threshold is not None and max_prob_label.probability < probability_threshold:
                continue
            color = max_prob_label.color_tuple
            line_thickness = 3
            shape = annotation.shape
            mask = self._add_shape_to_mask(
                shape=shape,
                mask=mask,
                labels=annotation.labels,
                color=color,
                line_thickness=line_thickness,
            )
        return mask

    def filter_by_confidence(self, confidence_threshold: float) -> "Prediction":
        """
        Return a new Prediction instance containing only those predicted annotations
        that have a confidence higher than `confidence_threshold`.

        :param confidence_threshold: Float between 0 and 1. Annotations that only
            have predicted labels with a probability lower than this value will be
            filtered out.
        :return: new Prediction object containing only annotations with a predicted
            probability higher than the confidence_threshold
        """
        annotations: list[Annotation] = []
        for annotation in self.annotations:
            max_prob = max([label.probability for label in annotation.labels])
            if max_prob > confidence_threshold:
                annotations.append(annotation)

        return Prediction(
            annotations=annotations,
            media_identifier=self.media_identifier,
            modified=datetime.now().isoformat(),
        )
