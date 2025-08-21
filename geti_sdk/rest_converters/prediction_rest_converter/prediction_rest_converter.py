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

import copy
from typing import Any

import attr

from geti_sdk.data_models import Annotation, Prediction
from geti_sdk.data_models.predictions import ResultMedium
from geti_sdk.data_models.utils import attr_value_serializer, remove_null_fields
from geti_sdk.rest_converters import AnnotationRESTConverter


class PredictionRESTConverter:
    """
    Class to convert REST representations of predictions into Prediction entities.
    """

    @staticmethod
    def from_dict(prediction: dict[str, Any]) -> Prediction:
        """
        Create a Prediction object from a dictionary returned by the
        /predictions REST endpoint in the Intel® Geti™ platform.

        :param prediction: dictionary representing a Prediction, which
            contains all prediction annotations for a certain media entity
        :return: Prediction object
        """
        input_copy = copy.deepcopy(prediction)
        annotations: list[Annotation] = []
        prediction_dicts = input_copy.pop("predictions", None)
        if prediction_dicts is None:
            # Geti versions lower than 1.13 still use 'annotations' as key
            prediction_dicts = prediction.get("annotations")
        for annotation in prediction_dicts:
            if not isinstance(annotation, Annotation):
                annotations.append(AnnotationRESTConverter.annotation_from_dict(annotation))
            else:
                annotations.append(annotation)
        media_identifier_dict = prediction.get("media_identifier")
        if media_identifier_dict is not None:
            media_identifier = AnnotationRESTConverter._media_identifier_from_dict(prediction["media_identifier"])
        else:
            media_identifier = None

        result_media: list[ResultMedium] = []
        for result_medium in prediction.get("maps", []):
            if not isinstance(result_medium, ResultMedium):
                result_media.append(ResultMedium(**result_medium))
            else:
                result_media.append(result_medium)
        input_copy.update(
            {
                "annotations": annotations,
                "media_identifier": media_identifier,
                "maps": result_media,
            }
        )
        return Prediction(**input_copy)

    @staticmethod
    def to_dict(prediction: Prediction, deidentify: bool = True) -> dict[str, Any]:
        """
        Convert a Prediction to a dictionary. By default, removes any ID
        fields in the output dictionary

        :param prediction: Prediction object to convert
        :param deidentify: True to remove any unique database ID fields in the output,
            False to keep these fields. Defaults to True
        :return: Dictionary holding the serialized Prediction data
        """
        if deidentify:
            prediction.deidentify()
        prediction_dict = attr.asdict(prediction, recurse=True, value_serializer=attr_value_serializer)
        remove_null_fields(prediction_dict)
        return prediction_dict
