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

import os
from unittest.mock import patch

import numpy as np

from geti_sdk.data_models import AnnotationScene, Image, VideoFrame
from geti_sdk.data_models.containers import MediaList
from geti_sdk.utils import (
    show_image_with_annotation_scene,
    show_video_frames_with_annotation_scenes,
)


class TestPlotHelpers:
    def test_show_image_with_annotation_scene_cv2(
        self,
        fxt_annotation_scene: AnnotationScene,
        fxt_numpy_image: np.ndarray,
        fxt_geti_image: Image,
        fxt_temp_directory: str,
    ):
        # Arrange
        filepath = os.path.join(fxt_temp_directory, "dummy_results.jpg")

        # Act
        with patch("geti_sdk.utils.plot_helpers.PILImage.Image.show") as mock_imshow:
            result = show_image_with_annotation_scene(image=fxt_numpy_image, annotation_scene=fxt_annotation_scene)
            result_geti = show_image_with_annotation_scene(image=fxt_geti_image, annotation_scene=fxt_annotation_scene)
            results_no_show = show_image_with_annotation_scene(
                image=fxt_geti_image,
                annotation_scene=fxt_annotation_scene,
                show_results=False,
            )
            results_filepath = show_image_with_annotation_scene(
                image=fxt_geti_image,
                annotation_scene=fxt_annotation_scene,
                filepath=filepath,
            )

        # Assert
        assert mock_imshow.call_count == 2
        assert result.shape == fxt_numpy_image.shape
        assert result_geti.shape == fxt_numpy_image.shape
        assert results_no_show.shape == fxt_numpy_image.shape
        assert results_filepath.shape == fxt_numpy_image.shape
        assert os.path.isfile(filepath)

    def test_show_video_frames_with_annotation_scenes(
        self,
        fxt_video_annotation_scenes: list[AnnotationScene],
        fxt_video_frames: MediaList[VideoFrame],
    ):
        # Act
        with (
            patch("cv2.imshow") as mock_imshow,
            patch("cv2.waitKey") as mock_waitkey,
            patch("cv2.destroyAllWindows") as mock_destroywindows,
        ):
            show_video_frames_with_annotation_scenes(
                video_frames=fxt_video_frames,
                annotation_scenes=fxt_video_annotation_scenes,
            )

        # Assert
        assert mock_imshow.call_count == len(fxt_video_frames)
        assert mock_waitkey.call_count == len(fxt_video_frames) + 1
        mock_destroywindows.assert_called_once()
