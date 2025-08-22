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
import math

import pytest

from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
)


class TestShapes:
    def test_rectangle(self, fxt_rectangle: Rectangle, fxt_rectangle_roi: Rectangle):
        # Arrange
        x_in_roi = fxt_rectangle_roi.x + fxt_rectangle.x
        y_in_roi = fxt_rectangle_roi.y + fxt_rectangle.y
        expected_area = fxt_rectangle.width * fxt_rectangle.height

        # Act
        rect_in_roi = fxt_rectangle.to_absolute_coordinates(parent_roi=fxt_rectangle_roi)
        area = fxt_rectangle.area

        # Assert
        assert fxt_rectangle.to_roi() == fxt_rectangle
        assert rect_in_roi.x == x_in_roi
        assert rect_in_roi.y == y_in_roi
        assert rect_in_roi.width == fxt_rectangle.width
        assert rect_in_roi.height == fxt_rectangle.height
        assert area == expected_area

    def test_ellipse(self, fxt_rectangle_roi: Rectangle, fxt_ellipse: Ellipse):
        # Arrange
        x_in_roi = fxt_rectangle_roi.x + fxt_ellipse.x
        y_in_roi = fxt_rectangle_roi.y + fxt_ellipse.y
        expected_area = fxt_ellipse.width * fxt_ellipse.height * math.pi
        expected_roi = Rectangle(
            x=fxt_ellipse.x,
            y=fxt_ellipse.y,
            width=fxt_ellipse.width,
            height=fxt_ellipse.height,
        )
        expected_xmax = fxt_ellipse.x + fxt_ellipse.width
        expected_ymax = fxt_ellipse.y + fxt_ellipse.height

        # Act
        ellipse_in_roi = fxt_ellipse.to_absolute_coordinates(parent_roi=fxt_rectangle_roi)
        area = fxt_ellipse.area
        roi = fxt_ellipse.to_roi()

        # Assert
        assert area == expected_area
        assert roi == expected_roi
        assert ellipse_in_roi.x == x_in_roi
        assert ellipse_in_roi.y == y_in_roi
        assert fxt_ellipse.x_max == expected_xmax
        assert fxt_ellipse.y_max == expected_ymax

    def test_polygon(self, fxt_triangle: Polygon, fxt_rectangle_roi: Rectangle):
        # Arrange
        x_roi, width_roi = 10, 40
        y_roi, height_roi = 20, 30
        expected_triangle_in_roi = Polygon(points=[Point(x=110, y=220), Point(x=130, y=250), Point(x=150, y=220)])
        expected_area = 0.5 * 30 * 40

        # Act
        area = fxt_triangle.area
        roi = fxt_triangle.to_roi()
        triangle_in_roi = fxt_triangle.to_absolute_coordinates(parent_roi=fxt_rectangle_roi)
        rotated_rect = fxt_triangle.fit_rotated_rectangle()

        # Assert
        assert area == expected_area
        assert roi.x == x_roi
        assert roi.y == y_roi
        assert roi.width == width_roi
        assert roi.height == height_roi
        assert triangle_in_roi == expected_triangle_in_roi
        assert rotated_rect.area == 1200

    def test_rotated_rectangle(
        self,
        fxt_rotated_rectangle: RotatedRectangle,
        fxt_rotated_rectangle_as_polygon: Polygon,
        fxt_rectangle_roi: Rectangle,
        fxt_triangle: Polygon,
    ):
        # Arrange
        expected_area = 5000
        expected_roi_area = 106 * 106
        x_in_roi = fxt_rectangle_roi.x + fxt_rotated_rectangle.x
        y_in_roi = fxt_rectangle_roi.y + fxt_rotated_rectangle.y

        # Act
        rotated_rect = RotatedRectangle.from_polygon(polygon=fxt_rotated_rectangle_as_polygon)
        roi = fxt_rotated_rectangle.to_roi()
        rotated_rect_in_roi = fxt_rotated_rectangle.to_absolute_coordinates(parent_roi=fxt_rectangle_roi)
        rotated_rect_polygon = fxt_rotated_rectangle.to_polygon()

        # Assert
        assert x_in_roi == rotated_rect_in_roi.x
        assert y_in_roi == rotated_rect_in_roi.y
        assert roi.area == expected_roi_area
        assert fxt_rotated_rectangle.area == expected_area
        assert rotated_rect.area == fxt_rotated_rectangle.area
        assert rotated_rect_polygon.area == fxt_rotated_rectangle_as_polygon.area

        with pytest.raises(ValueError):
            RotatedRectangle.from_polygon(polygon=fxt_triangle)

    def test_keypoint(self, fxt_keypoint: Keypoint, fxt_rectangle_roi: Rectangle):
        # Arrange
        expected_x = fxt_rectangle_roi.x + fxt_keypoint.x
        expected_y = fxt_rectangle_roi.y + fxt_keypoint.y
        expected_is_visible = True

        # Act
        keypoint_in_roi = fxt_keypoint.to_absolute_coordinates(parent_roi=fxt_rectangle_roi)

        # Assert
        assert keypoint_in_roi.x == expected_x
        assert keypoint_in_roi.y == expected_y
        assert keypoint_in_roi.is_visible == expected_is_visible

        with pytest.raises(NotImplementedError):
            fxt_keypoint.to_roi()
        with pytest.raises(NotImplementedError):
            _ = fxt_keypoint.area
