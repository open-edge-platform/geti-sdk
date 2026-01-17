# INTEL CONFIDENTIAL
#
# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
from unittest.mock import MagicMock

import numpy as np
import pytest
from model_api.models import (
    AnomalyResult,
    ClassificationResult,
    Contour,
    DetectedKeypoints,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
)

from geti_sdk.data_models.containers import LabelList
from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.label import Label, ScoredLabel
from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
)
from geti_sdk.deployment.predictions_postprocessing.results_converter.results_to_prediction_converter import (
    AnomalyToPredictionConverter,
    ClassificationToPredictionConverter,
    DetectionToPredictionConverter,
    KeypointDetectionToPredictionConverter,
    RotatedRectToPredictionConverter,
    SegmentationToPredictionConverter,
)


def coords_to_xmin_xmax_width_height(
    coords: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    "Convert bbox to xmin, ymin, width, height format"
    x1, y1, x2, y2 = coords
    return x1, y1, x2 - x1, y2 - y1


class TestInferenceResultsToPredictionConverter:
    def test_classification_to_prediction_converter(self, fxt_label_list_factory):
        # Arrange
        labels = fxt_label_list_factory(Domain.CLASSIFICATION)
        labels = labels.get_non_empty_labels()
        raw_prediction = ClassificationResult(
            top_labels=[(1, labels[1].name, 0.81)],
            raw_scores=[0.19, 0.81],
            saliency_map=None,
            feature_vector=None,
        )
        model_api_labels = [label.name for label in labels]

        # Act
        converter = ClassificationToPredictionConverter(labels, configuration={"labels": model_api_labels})
        prediction = converter.convert_to_prediction(raw_prediction, image_shape=(10, 10))

        # Assert
        assert [converter.get_label_by_idx(i) for i in range(len(labels))] == labels
        assert [converter.get_label_by_str(label.name) for label in labels] == labels
        assert len(prediction.annotations) == 1
        predicted_label = prediction.annotations[0].labels[0]
        assert predicted_label.name == labels[1].name
        assert predicted_label.probability == 0.81

    @pytest.mark.parametrize("use_ellipse_shapes", [True, False])
    def test_detection_to_prediction_converter(self, use_ellipse_shapes, fxt_label_list_factory):
        # Arrange
        labels = fxt_label_list_factory(Domain.DETECTION)
        coords = [12.0, 41.0, 12.5, 45.5]
        raw_prediction = DetectionResult(
            bboxes=np.array([coords]),
            labels=np.array([0]),
            scores=np.array([0.51]),
        )
        model_api_labels = [label.name for label in labels]

        # Act
        converter = DetectionToPredictionConverter(
            labels=labels,
            configuration={
                "use_ellipse_shapes": use_ellipse_shapes,
                "labels": model_api_labels,
            },
        )
        prediction = converter.convert_to_prediction(raw_prediction)

        # Assert
        assert [converter.get_label_by_idx(i) for i in range(len(labels))] == labels
        assert [converter.get_label_by_str(label.name) for label in labels] == labels
        assert len(prediction.annotations) == 1
        if use_ellipse_shapes:
            assert prediction.annotations[0].shape == Ellipse(*coords_to_xmin_xmax_width_height(coords))
        else:
            assert prediction.annotations[0].shape == Rectangle(*coords_to_xmin_xmax_width_height(coords))
        assert prediction.annotations[0].labels[0].probability == pytest.approx(0.51)
        assert prediction.annotations[0].labels[0].name == labels[0].name

    @pytest.mark.parametrize("use_ellipse_shapes", [True, False])
    def test_rotated_rect_to_prediction_converter(self, use_ellipse_shapes, fxt_label_list_factory):
        # Arrange
        labels = fxt_label_list_factory(Domain.ROTATED_DETECTION)
        coords = [1, 1, 4, 4]
        score = 0.51
        mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        raw_prediction = InstanceSegmentationResult(
            bboxes=np.array([coords]),
            labels=np.array([1]),
            masks=np.array([mask]),
            scores=np.array([score]),
            label_names=[""],
            saliency_map=None,
            feature_vector=None,
        )
        height, width = mask.shape
        metadata = {"original_shape": (height, width, 3)}
        model_api_labels = [label.name for label in labels]

        # Act
        converter = RotatedRectToPredictionConverter(
            labels,
            configuration={
                "use_ellipse_shapes": use_ellipse_shapes,
                "labels": model_api_labels,
            },
        )
        prediction = converter.convert_to_prediction(raw_prediction, metadata=metadata)

        # Assert
        assert [converter.get_label_by_idx(i) for i in range(len(labels))] == labels
        assert [converter.get_label_by_str(label.name) for label in labels] == labels
        assert len(prediction.annotations) == 1
        if use_ellipse_shapes:
            assert prediction.annotations[0].shape == Ellipse(*coords_to_xmin_xmax_width_height(coords))
        else:
            assert prediction.annotations[0].shape == RotatedRectangle.from_polygon(
                Polygon(
                    points=[
                        Point(x=1, y=1),
                        Point(x=3, y=1),
                        Point(x=3, y=3),
                        Point(x=1, y=3),
                    ]
                )
            )
        assert prediction.annotations[0].labels[0] == ScoredLabel.from_label(label=labels[1], probability=score)

    def test_segmentation_to_prediction_converter(self, fxt_label_list_factory):
        # Arrange
        labels = fxt_label_list_factory(Domain.SEGMENTATION)
        labels_str = [label.name for label in labels]
        result_image = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
            ]
        )
        soft_predictions = np.array(
            [
                [[0.9, 0.1, 0.1], [0.7, 0.1, 0.2], [0.9, 0.1, 0.1]],
                [[0.9, 0.0, 0.1], [0.9, 0.0, 0.1], [0.9, 0.0, 0.0]],
                [[0.2, 0.2, 0.6], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]],
            ]
        )
        raw_prediction = ImageResultWithSoftPrediction(
            resultImage=result_image,
            soft_prediction=soft_predictions,
            saliency_map=np.zeros_like(result_image),
            feature_vector=np.zeros((result_image.shape[0], result_image.shape[1], 2)),
        )
        expected_shape = Polygon(points=[Point(1, 1), Point(0, 3), Point(1, 3), Point(3, 3)])
        seg_model = MagicMock()
        seg_model.labels = labels_str
        model_api_labels = [label.name for label in labels]

        def get_contours(_: ImageResultWithSoftPrediction):
            return [
                Contour(
                    labels_str[0],
                    0.8,
                    shape=np.array([[(1, 1)], [(0, 3)], [(1, 3)], [(3, 3)]]),
                )
            ]

        seg_model.get_contours = get_contours

        # Act
        converter = SegmentationToPredictionConverter(
            labels, configuration={"labels": model_api_labels}, model=seg_model
        )
        prediction = converter.convert_to_prediction(raw_prediction)

        # Assert
        assert [converter.get_label_by_idx(i + 1) for i in range(len(labels))] == labels
        assert [converter.get_label_by_str(label.name) for label in labels] == labels
        assert len(prediction.annotations) == 1
        assert prediction.annotations[0].labels[0] == ScoredLabel.from_label(labels[0], probability=0.8)
        assert isinstance(prediction.annotations[0].shape, Polygon)
        for p1, p2 in zip(prediction.annotations[0].shape.points, expected_shape.points):
            assert p1.x == pytest.approx(p2.x, 0.01)
            assert p1.y == pytest.approx(p2.y, 0.01)

    @pytest.mark.parametrize(
        "domain",
        [
            Domain.ANOMALY_CLASSIFICATION,
            Domain.ANOMALY_SEGMENTATION,
            Domain.ANOMALY_DETECTION,
        ],
    )
    def test_anomaly_to_prediction_converter(self, domain, fxt_label_list_factory):
        # Arrange
        labels = fxt_label_list_factory(domain)
        normal_label = next(label for label in labels if not label.is_anomalous)
        anomalous_label = next(label for label in labels if label.is_anomalous)
        anomaly_map = np.ones((2, 2))
        pred_boxes = np.array([[2, 2, 4, 4]])
        pred_mask = np.ones((2, 2))
        raw_prediction = AnomalyResult(
            anomaly_map=anomaly_map,
            pred_boxes=pred_boxes,
            pred_mask=pred_mask,
            pred_label="Anomalous",
            pred_score=1.0,
        )

        # Act
        converter = AnomalyToPredictionConverter(
            labels, configuration={"domain": domain, "labels": ["Anomalous", "Normal"]}
        )
        prediction = converter.convert_to_prediction(raw_prediction, image_shape=anomaly_map.shape)

        # Assert
        assert converter.normal_label == normal_label
        assert converter.anomalous_label == anomalous_label
        assert len(prediction.annotations) == 1
        assert prediction.annotations[0].labels[0] == ScoredLabel.from_label(
            next(label for label in labels if label.is_anomalous), probability=1.0
        )
        if domain == Domain.ANOMALY_SEGMENTATION:
            assert prediction.annotations[0].shape == Polygon(
                points=[
                    Point(0.0, 0.0),
                    Point(0.0, 1.0),
                    Point(1.0, 1.0),
                    Point(1.0, 0.0),
                ]
            )
        elif domain == Domain.ANOMALY_DETECTION:
            assert prediction.annotations[0].shape == Rectangle(*coords_to_xmin_xmax_width_height(pred_boxes[0]))

    def test_keypoint_to_prediction_converter(self, fxt_label_list_factory):
        # Arrange
        labels = fxt_label_list_factory(Domain.KEYPOINT_DETECTION)
        predicted_keypoints = [[10, 10], [20, 20], [30, 30]]
        scores = [0.5, 0.6, 0.7]
        raw_prediction = DetectedKeypoints(
            keypoints=predicted_keypoints,
            scores=scores,
        )

        # Act
        converter = KeypointDetectionToPredictionConverter(
            labels,
            configuration={
                "domain": Domain.KEYPOINT_DETECTION,
                "labels": [label.name for label in labels],
            },
        )
        prediction = converter.convert_to_prediction(raw_prediction)

        # Assert
        assert converter.labels == labels
        assert len(prediction.annotations) == len(predicted_keypoints)
        for idx, annotation in enumerate(prediction.annotations):
            assert isinstance(annotation.shape, Keypoint)
            assert annotation.shape.is_visible
            assert annotation.shape.x == predicted_keypoints[idx][0]
            assert annotation.shape.y == predicted_keypoints[idx][1]
            assert annotation.labels[0] == ScoredLabel.from_label(labels[idx], probability=scores[idx])

    @pytest.mark.parametrize(
        "label_ids, label_names, predicted_labels, configuration",
        [
            (
                ["1", "2"],
                ["foo bar1", "foo_bar2"],
                ["foo_bar1", "foo_bar2"],
                {"label_ids": ["1", "2"], "labels": ["foo_bar1", "foo_bar2"]},
            ),
            (
                ["1", "2"],
                ["label 1", "label 2"],
                ["label_1", "label_2"],
                {"labels": ["label_1", "label_2"]},
            ),
            (["1", "2"], ["?", "@"], ["@", "?"], {"labels": ["?", "@"]}),
            (
                ["1", "2", "3", "4"],
                ["c", "b", "a", "empty"],
                ["a", "b", "c"],
                {"labels": ["c", "b", "a", "empty"]},
            ),
            (["1"], ["1"], ["1"], {"labels": [1]}),
        ],
    )
    def test_legacy_label_conversion(self, label_ids, label_names, predicted_labels, configuration):
        # Arrange
        labels = LabelList(
            [
                Label(
                    id=_id,
                    name=name,
                    color="",
                    group="",
                    domain=Domain.CLASSIFICATION,
                    is_empty="empty" in name,
                )
                for _id, name in zip(label_ids, label_names)
            ]
        )
        raw_prediction = ClassificationResult(
            top_labels=[(i, p_label, 0.7) for i, p_label in enumerate(predicted_labels)],
            raw_scores=[0.7] * len(predicted_labels),
            saliency_map=None,
            feature_vector=None,
        )

        # Act
        converter = ClassificationToPredictionConverter(labels=labels, configuration=configuration)
        pred = converter.convert_to_prediction(
            inference_results=raw_prediction,
            image_shape=(10, 10, 10),
        )

        # Assert
        assert len(pred.annotations[0].labels) == len(predicted_labels)
        assert {label.name for label in converter.labels} == {p_label.name for p_label in pred.annotations[0].labels}
