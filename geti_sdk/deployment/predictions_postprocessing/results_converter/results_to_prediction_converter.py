
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
"""Converters from Model API inference results to Geti SDK Prediction objects.

This module provides robust, version-tolerant conversion utilities for multiple
result shapes emitted by Model API across releases. It avoids ambiguous boolean
evaluation for NumPy arrays, performs early schema/shape validation, and offers
clear logging and error messages to aid maintainability and debugging.
"""

from __future__ import annotations

import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import cv2
import numpy as np
from model_api.models import (
    AnomalyResult,
    ClassificationResult,
    DetectedKeypoints,
    DetectionResult,
    ImageModel,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    SegmentationModel,
)

from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.containers import LabelList
from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.label import Label, ScoredLabel
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
)
from geti_sdk.deployment.predictions_postprocessing.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
)

LOGGER = logging.getLogger(__name__)


# ---------- Lightweight protocols for static expectations (optional but useful) ----------

class VectorizedSegResult(Protocol):
    """Vectorized instance segmentation result:
    - masks: (N, H, W) or a sequence of (H, W) arrays
    - labels: sequence of int
    - scores: sequence of float
    - bboxes: optional (N, 4), (x1, y1, x2, y2) per instance
    """
    masks: Sequence[np.ndarray]
    labels: Sequence[int]
    scores: Sequence[float]
    bboxes: Optional[Sequence[Sequence[float]]]


class ObjectListItem(Protocol):
    """Item of object-list result schema with flexible field names.
    Actual access is done via helper that searches canonical names."""
    # No attributes declared here intentionally; accessed via helper.


class ObjectListSegResult(Protocol):
    """Object-list instance segmentation result with either 'instances' or 'objects'."""
    instances: Sequence[ObjectListItem]  # may not exist; we treat via helper
    # or:
    objects: Sequence[ObjectListItem]


# ---------- Common base converter ----------

class InferenceResultsToPredictionConverter(metaclass=abc.ABCMeta):
    """Base interface for all converters with label mapping and configuration."""

    def __init__(self, labels: LabelList, configuration: Dict[str, Any]):
        self.labels = labels.get_non_empty_labels()

        # Optional behavior flags
        self.strict_schema: bool = configuration.get("strict_schema", True)

        # Model API label configuration
        model_api_labels = configuration["labels"]
        label_ids = configuration.get("label_ids", [])  # may be empty

        # Normalize 'labels' to list[string]
        model_api_labels = (
            model_api_labels.split()  # space-separated string
            if isinstance(model_api_labels, str)
            else [str(name) for name in model_api_labels]
        )

        # Maps
        self.label_map_ids: Dict[str, Label] = {}
        self.legacy_label_map_names: Dict[str, List[Label]] = defaultdict(list)
        self.empty_label: Label = labels.get_empty_label()

        # Populate maps from the LabelList
        for label in labels:
            self.label_map_ids[str(label.id)] = label
            # Handle both "foo bar" and "foo_bar"
            self.legacy_label_map_names[label.name.replace(" ", "_")].append(label)
            self.legacy_label_map_names[label.name].append(label)
        self.legacy_label_map_names["otx_empty_lbl"] = [self.empty_label]

        # Create mapping from Model API index/string → Label
        self.idx_to_label: Dict[int, Label] = {}
        self.str_to_label: Dict[str, Label] = {}
        self.model_api_label_map_counts: Dict[str, int] = defaultdict(int)

        # If label_ids are missing, generate placeholders so that zipping works
        n_missing_ids = len(model_api_labels) - len(label_ids)
        if n_missing_ids > 0:
            LOGGER.warning(
                "Mismatch between label_ids (len=%d) and model_api_labels (len=%d). "
                "Using placeholder label IDs for the missing %d labels.",
                len(label_ids),
                len(model_api_labels),
                n_missing_ids,
            )
            for i in range(n_missing_ids):
                label_ids.append(f"generated_label_{i}")

        for i, (label_id_str, label_str) in enumerate(zip(label_ids, model_api_labels)):
            try:
                label = self.__get_label(label_str, pos_idx=self.model_api_label_map_counts[label_str])
            except ValueError:
                # If the label has been renamed, try to resolve by ID
                if label_id_str in self.label_map_ids:
                    label = self.label_map_ids[label_id_str]
                    LOGGER.warning("Label '%s' has been renamed to '%s'.", label_str, label.name)
                else:
                    LOGGER.warning("Label '%s' cannot be found. It may have been removed.", label_str)
                    label = self.empty_label
            self.idx_to_label[i] = label
            self.str_to_label[label_str] = label
            self.model_api_label_map_counts[label_str] += 1

        LOGGER.info("Converter loaded labels with indices: %s", self.idx_to_label)

    def __get_label(self, label_str: str, pos_idx: int) -> Label:
        """Resolve label by ID-string or (legacy) name with position for duplicates."""
        if label_str in self.label_map_ids:
            return self.label_map_ids[label_str]
        matched = self.legacy_label_map_names[label_str]
        if pos_idx < len(matched):
            return matched[pos_idx]
        raise ValueError(f"Label '{label_str}' (pos_idx={pos_idx}) not found in the label schema")

    def get_label_by_idx(self, label_idx: int) -> Label:
        """Return a Label object by its Model API index."""
        return self.idx_to_label[label_idx]

    def get_label_by_str(self, label_str: str) -> Label:
        """Return a Label object by its Model API string name."""
        return self.str_to_label[label_str]

    @abc.abstractmethod
    def convert_to_prediction(self, inference_results: NamedTuple, **kwargs) -> Prediction:
        """Convert raw inference results to a Geti Prediction."""
        raise NotImplementedError

    @abc.abstractmethod
    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        **kwargs,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract a saliency map from inference results in a unified format."""
        raise NotImplementedError


# ---------- Classification ----------

class ClassificationToPredictionConverter(InferenceResultsToPredictionConverter):
    """Convert Model API Classification results to Prediction."""

    def __init__(self, labels: LabelList, configuration: Dict[str, Any]):
        super().__init__(labels, configuration)

    def convert_to_prediction(
        self,
        inference_results: ClassificationResult,
        image_shape: Tuple[int, int, int],
        **kwargs,
    ) -> Prediction:
        labels: List[ScoredLabel] = []
        for label in inference_results.top_labels:
            label_idx, _, label_prob = label
            scored_label = ScoredLabel.from_label(
                label=self.get_label_by_idx(label_idx),
                probability=label_prob,
            )
            labels.append(scored_label)

        if not labels and self.empty_label:
            labels = [ScoredLabel.from_label(self.empty_label, probability=0)]

        annotations = Annotation(
            shape=Rectangle.generate_full_box(image_shape[1], image_shape[0]),
            labels=labels,
        )
        return Prediction([annotations])

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        saliency_map = getattr(inference_results, "saliency_map", None)
        if saliency_map is None or len(saliency_map) == 0:
            return None

        # Expected shape: (N, H, W) after transpose/squeeze
        smap = np.transpose(saliency_map.squeeze(0), (1, 2, 0))
        smap = cv2.resize(smap, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        if smap.ndim == 2:
            smap = np.expand_dims(smap, axis=-1)
        smap = np.transpose(smap, (2, 0, 1))
        return {label.name: smap[i] for i, label in enumerate(self.labels.get_non_empty_labels())}


# ---------- Detection ----------

class DetectionToPredictionConverter(InferenceResultsToPredictionConverter):
    """Convert Model API Detection results to Prediction."""

    def __init__(self, labels: LabelList, configuration: Dict[str, Any]):
        super().__init__(labels, configuration)
        self.use_ellipse_shapes: bool = configuration.get("use_ellipse_shapes", False)
        self.confidence_threshold: float = configuration.get("confidence_threshold", 0.0)

    def _detection2array(self, detection: DetectionResult) -> np.ndarray:
        """Convert DetectionResult to numpy array [label, score, x1, y1, x2, y2]."""
        valid = []
        for score, label, bbox in zip(detection.scores, detection.labels, detection.bboxes):
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 1.0:
                valid.append((score, label, bbox))

        if not valid:
            return np.empty((0, 6), dtype=np.float32)

        result = np.empty((len(valid), 6), dtype=np.float32)
        for i, (score, label, bbox) in enumerate(valid):
            result[i] = [label, score, bbox[0], bbox[1], bbox[2], bbox[3]]
        return result

    def convert_to_prediction(self, inference_results: DetectionResult, **kwargs) -> Prediction:
        detections = self._detection2array(inference_results)
        annotations: List[Annotation] = []

        if len(detections) and (detections.shape[1:] < (6,) or detections.shape[1:] > (7,)):
            raise ValueError(
                f"Unexpected prediction shape; expected (n, 7) or (n, 6) but got {detections.shape}"
            )

        for detection in detections:
            # Some models use output [7,], where first element is not used
            _detection = detection[1:] if detection.shape == (7,) else detection
            label_index = int(_detection[0])
            confidence = float(_detection[1])

            if confidence < self.confidence_threshold:
                continue

            scored_label = ScoredLabel.from_label(self.get_label_by_idx(label_index), confidence)
            x1, y1, x2, y2 = _detection[2:]
            width, height = (x2 - x1), (y2 - y1)

            if self.use_ellipse_shapes:
                shape = Ellipse(float(x1), float(y1), float(width), float(height))
            else:
                shape = Rectangle(float(x1), float(y1), float(width), float(height))

            annotations.append(Annotation(shape=shape, labels=[scored_label]))

        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        saliency_map = getattr(inference_results, "saliency_map", None)
        if saliency_map is None or len(saliency_map) == 0:
            return None

        if isinstance(saliency_map, list):
            smap = np.array(
                [
                    s if len(s) > 0 else np.zeros(image_shape[:2], dtype=np.uint8)
                    for s in saliency_map
                ]
            )
        elif isinstance(saliency_map, np.ndarray):
            smap = saliency_map.squeeze(0)
        else:
            raise ValueError(f"Unsupported saliency_map type: {type(saliency_map)}")

        smap = cv2.resize(np.transpose(smap, (1, 2, 0)), dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        if smap.ndim == 2:
            smap = np.expand_dims(smap, axis=-1)
        smap = np.transpose(smap, (2, 0, 1))
        return {label.name: smap[i] for i, label in enumerate(self.labels)}


# ---------- Rotated detection (polygonization from masks) ----------

class RotatedRectToPredictionConverter(DetectionToPredictionConverter):
    """Convert instance segmentation results into rotated rectangles (polygon → RotatedRectangle)."""

    def convert_to_prediction(self, inference_results: InstanceSegmentationResult, **kwargs) -> Prediction:
        annotations: List[Annotation] = []

        for bbox, label_idx, mask, score in zip(
            inference_results.bboxes,
            inference_results.labels,
            inference_results.masks,
            inference_results.scores,
        ):
            label = self.get_label_by_idx(int(label_idx))
            if float(score) < self.confidence_threshold or label.is_empty:
                continue

            if self.use_ellipse_shapes:
                x1, y1, x2, y2 = bbox
                shape = Ellipse(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                annotations.append(
                    Annotation(shape=shape, labels=[ScoredLabel.from_label(label, float(score))])
                )
                continue

            mask = np.asarray(mask, dtype=np.uint8)
            contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchies is None:
                continue

            for contour, hierarchy in zip(contours, hierarchies[0]):
                # Only top-level contours
                if hierarchy[3] != -1:
                    continue
                if len(contour) <= 2 or cv2.contourArea(contour) < 1.0:
                    continue

                points = [Point(x=p[0], y=p[1]) for p in cv2.boxPoints(cv2.minAreaRect(contour))]
                poly = Polygon(points=points)
                shape = RotatedRectangle.from_polygon(poly)
                annotations.append(
                    Annotation(shape=shape, labels=[ScoredLabel.from_label(label, float(score))])
                )

        return Prediction(annotations)


# ---------- Instance segmentation (robust multi-schema handling) ----------

class MaskToAnnotationConverter(DetectionToPredictionConverter):
    """Convert instance segmentation results of various schemas to Prediction.

    Supported input schemas:
      1) Legacy:     `segmentedObjects`
      2) Vectorized: `masks`, `labels`, `scores` (+ optional `bboxes`)
      3) Object-list: `instances` or `objects` (items may expose different field names)

    All inputs are normalized to `_InstanceData` before geometry filtering.
    Raises:
      TypeError  - unsupported schema
      ValueError - inconsistent shapes (e.g., labels vs scores length)
    """

    # ---------------- Schema predication helpers ----------------

    @staticmethod
    def _is_legacy_segmented_objects(res: Any) -> bool:
        return hasattr(res, "segmentedObjects")

    @staticmethod
    def _is_vectorized_result(res: Any) -> bool:
        # Strict presence of vectorized components
        return all(hasattr(res, a) for a in ("masks", "labels", "scores"))

    @staticmethod
    def _is_object_list_result(res: Any) -> bool:
        return any(hasattr(res, a) for a in ("instances", "objects"))

    # ---------------- Normalization helpers ----------------

    @dataclass(frozen=True)
    class _InstanceData:
        label_value: Optional[int | str]
        score: float
        mask: Optional[np.ndarray]
        contours: Optional[Any]
        bbox: Optional[Tuple[float, float, float, float]]

    @staticmethod
    def _first_attr(obj: Any, names: Tuple[str, ...]) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _legacy_instance_data(self, instance: Any) -> "_InstanceData":
        return self._InstanceData(
            label_value=getattr(instance, "id", None),
            score=float(getattr(instance, "score", 0.0)),
            mask=getattr(instance, "mask", None),
            contours=None,
            bbox=(
                float(getattr(instance, "xmin", 0.0)),
                float(getattr(instance, "ymin", 0.0)),
                float(getattr(instance, "xmax", 0.0)),
                float(getattr(instance, "ymax", 0.0)),
            ),
        )

    def _new_instance_data(self, instance: Any) -> Optional["_InstanceData"]:
        label_value = self._first_attr(instance, ("label_id", "label", "label_index", "class_id", "category_id", "id"))
        if label_value is None:
            return None

        score = self._first_attr(instance, ("score", "confidence", "probability"))
        score_f = float(score) if score is not None else 0.0

        mask = self._first_attr(instance, ("mask", "segmentation"))
        contours = self._first_attr(instance, ("contours", "contour"))
        bbox = self._first_attr(instance, ("bbox", "box"))
        if bbox is not None and len(bbox) == 4:
            bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        else:
            bbox = None

        return self._InstanceData(
            label_value=label_value,
            score=score_f,
            mask=np.array(mask) if mask is not None else None,
            contours=contours,
            bbox=bbox,
        )

    def _resolve_label(self, label_value: Optional[int | str]) -> Optional[Label]:
        """Resolve label by id or string; supports legacy string names (spaces vs underscores)."""
        if label_value is None:
            return None

        # String name resolution (including legacy)
        if isinstance(label_value, str):
            try:
                return self.get_label_by_str(label_value)
            except KeyError:
                legacy = label_value.replace(" ", "_")
                try:
                    LOGGER.debug("Resolving label by legacy string '%s' -> '%s'.", label_value, legacy)
                    return self.get_label_by_str(legacy)
                except KeyError:
                    LOGGER.warning("Unknown string label: '%s'.", label_value)
                    return None

        # Integer/ID resolution
        try:
            return self.get_label_by_idx(int(label_value))
        except Exception:
            LOGGER.warning("Unable to resolve label by index: %s", label_value)
            return None

    @staticmethod
    def _ensure_bbox_from_instance(instance: "_InstanceData") -> Optional[Tuple[float, float, float, float]]:
        if instance.bbox is not None:
            return instance.bbox
        if instance.mask is None:
            return None

        coords = np.column_stack(np.where(instance.mask > 0))
        if coords.size == 0:
            return None
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

    @staticmethod
    def _contour_pairs_from_instance(instance: "_InstanceData") -> Optional[Iterable[Tuple[np.ndarray, List[int]]]]:
        if instance.mask is not None:
            mask = instance.mask.astype(np.uint8)
            contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchies is None:
                # Treat all contours as top-level in fallback
                return ((c, [0, 0, 0, -1]) for c in contours)
            return zip(contours, hierarchies[0])

        if instance.contours is not None:
            contours = instance.contours
            if isinstance(contours, np.ndarray):
                contours = [contours]
            return ((cnt, [0, 0, 0, -1]) for cnt in contours)

        return None

    @staticmethod
    def _normalize_bboxes_to_tuple_list(bboxes: Any) -> Optional[List[Tuple[float, float, float, float]]]:
        if bboxes is None:
            return None
        if isinstance(bboxes, np.ndarray):
            if bboxes.ndim != 2 or bboxes.shape[1] != 4:
                raise ValueError(f"Expected bboxes with shape (N, 4), got {bboxes.shape}")
            return [tuple(float(x) for x in row) for row in bboxes]
        if isinstance(bboxes, (list, tuple)):
            out: List[Tuple[float, float, float, float]] = []
            for bb in bboxes:
                if bb is None:
                    out.append(None)  # type: ignore[arg-type]
                elif len(bb) == 4:
                    out.append(tuple(float(x) for x in bb))  # type: ignore[arg-type]
                else:
                    raise ValueError(f"Expected bbox of len 4, got {len(bb)}")
            return out
        raise ValueError(f"Unsupported bboxes type: {type(bboxes)}")

    # ---------------- Conversion ----------------

    def convert_to_prediction(self, inference_results: Any, **kwargs: Dict[str, Any]) -> Prediction:
        logger = LOGGER
        annotations: List[Annotation] = []

        # --- Schema dispatch with explicit predicates ---
        if self._is_legacy_segmented_objects(inference_results):
            logger.debug("MaskToAnnotationConverter: using legacy segmentedObjects results.")
            instances_iter = (
                self._legacy_instance_data(obj) for obj in inference_results.segmentedObjects
            )

        elif self._is_vectorized_result(inference_results):
            logger.debug("MaskToAnnotationConverter: using vectorized arrays (masks/labels/scores).")

            vec_masks = getattr(inference_results, "masks", None)
            vec_labels = getattr(inference_results, "labels", None)
            vec_scores = getattr(inference_results, "scores", None)
            vec_bboxes = getattr(inference_results, "bboxes", None)

            # Normalize None → empty sequences (avoid boolean checks on ndarrays)
            vec_masks = [] if vec_masks is None else vec_masks
            vec_labels = [] if vec_labels is None else vec_labels
            vec_scores = [] if vec_scores is None else vec_scores
            norm_bboxes = self._normalize_bboxes_to_tuple_list(vec_bboxes)

            # Early shape checks
            if isinstance(vec_masks, np.ndarray):
                # Expecting (N, H, W)
                if vec_masks.ndim != 3:
                    raise ValueError(f"Expected masks as (N,H,W), got {vec_masks.shape}")
                n = vec_masks.shape[0]
                masks_len = n
                get_mask = lambda i: vec_masks[i]
            else:
                masks_len = len(vec_masks)
                get_mask = lambda i: vec_masks[i] if i < masks_len else None

            labels_len = len(vec_labels)
            scores_len = len(vec_scores)
            if labels_len != scores_len:
                raise ValueError(f"labels/scores length mismatch: {labels_len} vs {scores_len}")

            if norm_bboxes is not None and len(norm_bboxes) not in (0, labels_len):
                raise ValueError("bboxes length must be 0 or match labels/scores length")

            def _yield_vector_instances() -> Iterable[MaskToAnnotationConverter._InstanceData]:
                for i, (lbl, scr) in enumerate(zip(vec_labels, vec_scores)):
                    mask = get_mask(i)
                    bbox = None
                    if norm_bboxes is not None and i < len(norm_bboxes):
                        bbox = norm_bboxes[i]
                    yield MaskToAnnotationConverter._InstanceData(
                        label_value=int(lbl),
                        score=float(scr),
                        mask=np.array(mask) if mask is not None else None,
                        contours=None,
                        bbox=bbox,
                    )

            instances_iter = _yield_vector_instances()

        elif self._is_object_list_result(inference_results):
            logger.debug("MaskToAnnotationConverter: using object-list (instances/objects) results.")
            candidates = (
                self._first_attr(inference_results, ("instances", "objects")) or []
            )
            instances_iter = (
                data
                for instance in candidates
                if (data := self._new_instance_data(instance)) is not None
            )
        else:
            msg = (
                "Unsupported InstanceSegmentationResult schema: expected "
                "legacy(segmentedObjects) | vectorized(masks/labels/scores) | object-list(instances/objects)."
            )
            logger.error(msg)
            if self.strict_schema:
                raise TypeError(msg)
            # Non-strict fallback: produce empty prediction
            return Prediction([])

        # Materialize instances for deterministic metrics/logging
        instances: List[MaskToAnnotationConverter._InstanceData] = list(instances_iter)

        # --- Filtering and shape conversion ---
        total_candidates = len(instances)
        after_score = 0
        after_label = 0
        after_geometry = 0

        for instance in instances:
            if float(instance.score) < self.confidence_threshold:
                continue
            after_score += 1

            label = self._resolve_label(instance.label_value)
            if label is None or label.is_empty:
                continue
            after_label += 1

            if self.use_ellipse_shapes:
                bbox = self._ensure_bbox_from_instance(instance)
                if bbox is None:
                    continue
                xmin, ymin, xmax, ymax = bbox
                shape = Ellipse(
                    float(xmin),
                    float(ymin),
                    float(xmax - xmin),
                    float(ymax - ymin),
                )
                annotations.append(
                    Annotation(
                        shape=shape,
                        labels=[ScoredLabel.from_label(label, float(instance.score))],
                    )
                )
                after_geometry += 1
                continue

            contour_pairs = self._contour_pairs_from_instance(instance)
            if contour_pairs is None:
                continue

            for contour, hierarchy in contour_pairs:
                # Only top-level contours
                if hierarchy[3] != -1:
                    continue
                if len(contour) <= 2 or cv2.contourArea(contour) < 1.0:
                    continue

                contour = np.asarray(contour)
                if contour.ndim == 3 and contour.shape[1] == 1:
                    contour = contour[:, 0, :]

                points = [Point(x=float(pt[0]), y=float(pt[1])) for pt in contour]
                shape = Polygon(points=points)
                annotations.append(
                    Annotation(
                        shape=shape,
                        labels=[ScoredLabel.from_label(label, float(instance.score))],
                    )
                )
                after_geometry += 1

        LOGGER.debug(
            "[mask2ann] candidates=%d, after_score=%d, after_label=%d, "
            "after_geometry=%d, final_shapes=%d",
            total_candidates,
            after_score,
            after_label,
            after_geometry,
            len(annotations),
        )

        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        saliency_map = getattr(inference_results, "saliency_map", None)
        if saliency_map is None or len(saliency_map) == 0:
            return None

        # Model API returns list[np.ndarray] per class (including 'no_object' which may be empty)
        tmp = np.array(
            [
                sm if len(sm) > 0 else np.zeros(image_shape[:2], dtype=np.uint8)
                for sm in saliency_map
            ]
        )
        # shape: (N classes, H, W)
        return {label.name: tmp[i] for i, label in enumerate(self.labels)}


# ---------- Semantic segmentation ----------

class SegmentationToPredictionConverter(InferenceResultsToPredictionConverter):
    """Convert Model API semantic segmentation results to Prediction."""

    def __init__(self, labels: LabelList, configuration: Dict[str, Any], model: SegmentationModel):
        super().__init__(labels, configuration)
        self.model = model

    def get_label_by_idx(self, label_idx: int) -> Label:
        """Override: index=0 is reserved for 'background'."""
        self.idx_to_label[-1] = self.empty_label
        return super().get_label_by_idx(label_idx - 1)

    def convert_to_prediction(
        self,
        inference_results: ImageResultWithSoftPrediction,
        **kwargs,
    ) -> Prediction:
        contours = self.model.get_contours(inference_results)
        annotations: List[Annotation] = []

        for contour in contours:
            label = self.get_label_by_str(contour.label)
            if len(contour.shape) > 0 and not label.is_empty:
                approx_curve = cv2.approxPolyDP(contour.shape, 1.0, True)
                if len(approx_curve) > 2:
                    points = [Point(x=p[0][0], y=p[0][1]) for p in contour.shape]
                    annotations.append(
                        Annotation(
                            shape=Polygon(points=points),
                            labels=[
                                ScoredLabel.from_label(
                                    label=label, probability=float(contour.probability)
                                )
                            ],
                        )
                    )
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        saliency_map = getattr(inference_results, "saliency_map", None)
        if saliency_map is None or len(saliency_map) == 0:
            return None
        smap = np.transpose(saliency_map, (2, 0, 1))  # shape: (N classes, H, W)
        return {label.name: smap[i] for i, label in self.idx_to_label.items() if not label.is_empty}


# ---------- Anomaly ----------

class AnomalyToPredictionConverter(InferenceResultsToPredictionConverter):
    """Convert Model API Anomaly results to Prediction."""

    def __init__(self, labels: LabelList, configuration: Dict[str, Any]):
        # Exactly one normal and one anomalous label are expected
        self.normal_label = next(label for label in labels if not label.is_anomalous)
        self.anomalous_label = next(label for label in labels if label.is_anomalous)
        if configuration is not None and "domain" in configuration:
            self.domain = configuration["domain"]

    def convert_to_prediction(
        self,
        inference_results: AnomalyResult,
        image_shape: Tuple[int, int, int],
        **kwargs,
    ) -> Prediction:
        pred_label = getattr(inference_results, "pred_label", "")
        label = self.anomalous_label if str(pred_label).lower() in ("anomaly", "anomalous") else self.normal_label
        annotations: List[Annotation] = []

        if self.domain == Domain.ANOMALY_CLASSIFICATION or self.domain == Domain.ANOMALY:
            scored_label = ScoredLabel.from_label(
                label=label,
                probability=float(inference_results.pred_score),
            )
            annotations = [
                Annotation(
                    shape=Rectangle.generate_full_box(*image_shape[1::-1]),
                    labels=[scored_label],
                )
            ]
        elif self.domain == Domain.ANOMALY_SEGMENTATION:
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=inference_results.pred_mask,
                soft_prediction=inference_results.anomaly_map.squeeze(),
                label_map={0: self.normal_label, 1: self.anomalous_label},
            )
        elif self.domain == Domain.ANOMALY_DETECTION:
            for box in inference_results.pred_boxes:
                annotations.append(
                    Annotation(
                        shape=Rectangle(box[0], box[1], box[2] - box[0], box[3] - box[1]),
                        labels=[
                            ScoredLabel.from_label(
                                label=self.anomalous_label,
                                probability=float(inference_results.pred_score),
                            )
                        ],
                    )
                )
        else:
            raise ValueError(
                f"Cannot convert inference results for task '{self.domain.name}'. Only Anomaly tasks are supported."
            )

        if not annotations:
            scored_label = ScoredLabel.from_label(label=self.normal_label, probability=0)
            annotations = [
                Annotation(
                    labels=[scored_label],
                    shape=Rectangle.generate_full_box(*image_shape[1::-1]),
                )
            ]
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        # Normalized anomaly map: [0..255] uint8
        saliency_map = np.array(getattr(inference_results, "anomaly_map", None))
        if saliency_map is None:
            return None
        saliency_map = saliency_map - saliency_map.min()
        denom = (saliency_map.max() + 1e-12)
        saliency_map = (saliency_map / denom) * 255.0
        saliency_map = np.round(saliency_map).astype(np.uint8)
        return {self.anomalous_label.name: saliency_map}


# ---------- Keypoint detection ----------

class KeypointDetectionToPredictionConverter(InferenceResultsToPredictionConverter):
    """Convert Model API keypoint detection results to Prediction."""

    def __init__(self, labels: LabelList, configuration: Dict[str, Any]):
        super().__init__(labels, configuration)

    def convert_to_prediction(self, inference_results: DetectedKeypoints, **kwargs) -> Prediction:
        annotations: List[Annotation] = []
        for label_idx, (keypoint, score) in enumerate(zip(inference_results.keypoints, inference_results.scores)):
            shape = Keypoint(x=float(keypoint[0]), y=float(keypoint[1]), is_visible=True)
            label = self.get_label_by_idx(label_idx=label_idx)
            scored_label = ScoredLabel.from_label(label=label, probability=float(score))
            annotations.append(Annotation(shape=shape, labels=[scored_label]))
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        raise NotImplementedError


# ---------- Factory ----------

class ConverterFactory:
    """Factory for creating inference result → prediction converters per domain."""

    @staticmethod
    def create_converter(
        labels: LabelList,
        domain: Domain,
        configuration: Dict[str, Any],
        model: ImageModel,
    ) -> InferenceResultsToPredictionConverter:
        if domain == Domain.CLASSIFICATION:
            return ClassificationToPredictionConverter(labels, configuration)
        if domain == Domain.DETECTION:
            return DetectionToPredictionConverter(labels, configuration)
        if domain == Domain.SEGMENTATION:
            return SegmentationToPredictionConverter(labels, configuration, model=model)  # type: ignore[arg-type]
        if domain == Domain.ROTATED_DETECTION:
            return RotatedRectToPredictionConverter(labels, configuration)
        if domain == Domain.INSTANCE_SEGMENTATION:
            return MaskToAnnotationConverter(labels, configuration)
        if domain == Domain.KEYPOINT_DETECTION:
            return KeypointDetectionToPredictionConverter(labels, configuration)
        if domain in (
            Domain.ANOMALY_CLASSIFICATION,
            Domain.ANOMALY_SEGMENTATION,
            Domain.ANOMALY_DETECTION,
            Domain.ANOMALY,
        ):
            configuration.update({"domain": domain})
            return AnomalyToPredictionConverter(labels, configuration)

        raise ValueError(f"Cannot create converter for domain '{domain.name}'.")
