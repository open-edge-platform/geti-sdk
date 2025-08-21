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
from PIL import Image as PILImage

from geti_sdk.data_models.annotation_scene import AnnotationScene
from geti_sdk.data_models.containers import MediaList
from geti_sdk.data_models.enums import AnnotationKind
from geti_sdk.data_models.media import Image, VideoFrame
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.prediction_visualization.visualizer import Visualizer


def show_image_with_annotation_scene(
    image: Image | VideoFrame | np.ndarray,
    annotation_scene: AnnotationScene | Prediction,
    filepath: str | None = None,
    show_in_notebook: bool = False,
    show_results: bool = True,
    channel_order: str = "rgb",
    show_labels: bool = True,
    show_confidences: bool = True,
    fill_shapes: bool = True,
) -> np.ndarray:
    """
    Display an image with an annotation_scene overlayed on top of it.

    :param image: Image to show prediction for.
    :param annotation_scene: Annotations or Predictions to overlay on the image
    :param filepath: Optional filepath to save the image with annotation overlay to.
        If left as None, the result will not be saved to file
    :param show_in_notebook: True if the image needs to be shown in a notebook context.
        Setting this to True will display the image inline in the notebook. Setting it
        to False will open a pop up to show the image.
    :param show_results: True to show the results. If `show_in_notebook` is True, this
        will display the image with the annotations inside the notebook. If
        `show_in_notebook` is False, a new opencv window will pop up. If
        `show_results` is set to False, the results will not be shown but will only
        be returned instead
    :param channel_order: The channel order (R,G,B or B,G,R) used for the input image.
        This parameter accepts either `rgb` or `bgr` as input values, and defaults to
        `rgb`.
    :param show_labels: True to show the labels of the annotations. If set to False,
        the labels will not be shown.
    :param show_confidences: True to show the confidences of the annotations. If set to
        False, the confidences will not be shown.
    :param fill_shapes: True to fill the shapes of the annotations. If set to False, the
        shapes will not be filled.
    """
    if annotation_scene.kind == AnnotationKind.ANNOTATION:
        plot_type = "Annotation"
    elif annotation_scene.kind == AnnotationKind.PREDICTION:
        plot_type = "Prediction"
    else:
        raise ValueError(f"Invalid input: Unable to plot object of type {type(annotation_scene)}.")
    name = "Numpy image" if isinstance(image, np.ndarray) else image.name

    window_name = f"{plot_type} for {name}"
    visualizer = Visualizer(
        window_name=window_name,
        show_labels=show_labels,
        show_confidence=show_confidences,
    )

    numpy_image = image.copy() if isinstance(image, np.ndarray) else image.numpy.copy()

    if channel_order == "bgr":
        rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    elif channel_order == "rgb":
        rgb_image = numpy_image
    else:
        raise ValueError(f"Invalid channel order '{channel_order}'. Please use either `rgb` or `bgr`.")

    result = visualizer.draw(image=rgb_image, annotation=annotation_scene, fill_shapes=fill_shapes)

    # For compatibility with the previous version of the function
    # return image in BGR order; to be changed in 2.0.
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    if filepath is None:
        if show_results:
            image = PILImage.fromarray(result)
            if not show_in_notebook:
                image.show(title=window_name)
            else:
                try:
                    from IPython.display import display
                except ImportError as exc:
                    raise ImportError(
                        "IPython is required to display images in a notebook. "
                        "Please install the SDK with the 'notebooks' extra: 'geti-sdk[notebooks]'"
                    ) from exc
                display(image)
    else:
        success, buffer = cv2.imencode(".jpg", result_bgr)
        if success:
            buffer.tofile(filepath)
        else:
            raise RuntimeError("Unable to encode output image to .jpg format.")

    return result_bgr


def show_video_frames_with_annotation_scenes(
    video_frames: MediaList[VideoFrame],
    annotation_scenes: list[AnnotationScene | Prediction],
    wait_time: float = 1,
    filepath: str | None = None,
    show_labels: bool = True,
    show_confidences: bool = True,
    fill_shapes: bool = True,
):
    """
    Display a list of VideoFrames, with their annotations or predictions overlayed on
    top. The parameter `wait_time` specifies the time each frame is shown, in seconds.

    :param video_frames: List of VideoFrames to show
    :param annotation_scenes: List of AnnotationsScenes or Predictions to overlay on
        the frames
    :param wait_time: Time to show each frame, in seconds
    :param filepath: Optional filepath to save the video with annotation overlay to.
        If left as None, the video frames will be shown in a new opencv window
    :param show_labels: True to show the labels of the annotations. If set to False,
        the labels will not be shown.
    :param show_confidences: True to show the confidences of the annotations. If set to
        False, the confidences will not be shown.
    :param fill_shapes: True to fill the shapes of the annotations. If set to False, the
        shapes will not be filled.
    """
    first_frame = video_frames[0]

    out_writer: cv2.VideoWriter | None = None
    if filepath is not None:
        out_writer = cv2.VideoWriter(
            filename=f"{filepath}",
            fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps=1 / wait_time,
            frameSize=(
                first_frame.media_information.width,
                first_frame.media_information.height,
            ),
        )

    if annotation_scenes[0].kind == AnnotationKind.ANNOTATION:
        name = "Annotation"
    elif annotation_scenes[0].kind == AnnotationKind.PREDICTION:
        name = "Prediction"
    else:
        raise ValueError(f"Invalid input: Unable to plot object of type {type(annotation_scenes[0])}.")
    window_name = f"{name} for '{video_frames[0].video_name}'"
    visualizer = Visualizer(
        window_name=window_name,
        show_labels=show_labels,
        show_confidence=show_confidences,
    )

    for frame, annotation_scene in zip(video_frames, annotation_scenes):
        numpy_frame = frame.numpy.copy()
        result = visualizer.draw(numpy_frame, annotation=annotation_scene, fill_shapes=fill_shapes)

        if out_writer is None:
            cv2.imshow(window_name, result)
            cv2.waitKey(int(wait_time * 1000))
        else:
            out_writer.write(result)

    if out_writer is None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        out_writer.release()


def pad_image_and_put_caption(
    image: np.ndarray,
    run_name: int,
    model_1: str,
    model_1_score: str,
    model_2: str | None = None,
    model_2_score: str | None = None,
    fps: int | None = None,
) -> np.ndarray:
    """
    Pad the image with white and put the caption on it.

    :param image: Numpy array containing the image to be padded.
    :param run_name: Experiment description.
    :param model_1: Name of the model 1.
    :param model_1_score: Score of the model 1.
    :param model_2: Name of the model 2.
    :param model_2_score: Score of the model 2.
    :param fps: FPS of the inference.
    :return: Padded image with caption.
    """
    # Calculate text and image padding size
    text_scale = round(image.shape[1] / 1280, 1)
    thickness = int(text_scale / 1.5)
    (_, label_height), baseline = cv2.getTextSize("Test caption", cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
    universal_padding = 2
    bottom_padding_pre_line = label_height + baseline
    # Prepare image captions
    caption_lines = [
        run_name + ("" if fps is None else f" @{fps} fps"),
        f"Model 1: {model_1}, score {model_1_score:.2f}",
    ]
    if model_2 and model_2_score:
        caption_lines.append(f"Model 2: {model_2}, score {model_2_score:.2f}")
    # Pad the image and put captions on it
    padded_image = cv2.copyMakeBorder(
        image,
        top=universal_padding,
        bottom=universal_padding + bottom_padding_pre_line * len(caption_lines),
        left=universal_padding,
        right=universal_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    # Put text
    for line_number, text_line in enumerate(caption_lines):
        cv2.putText(
            padded_image,
            text_line,
            (0, image.shape[0] + bottom_padding_pre_line * (line_number + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            thickness,
        )
    return padded_image


def concat_prediction_results(results: list[list[np.ndarray]]) -> np.ndarray:
    """
    Merge the prediction images to one.

    :param results: List of lists of numpy arrays containing the results of the
        predictions.
    :return: Numpy array containing the concatenated results.
    """
    # Gather information about images on the grid
    row_pixel_lengths = []
    for index, row in enumerate(results):
        integral_row_length = sum([image.shape[1] for image in row])
        row_pixel_lengths.append(integral_row_length)
        image_heights = [image.shape[0] for image in row]
        if len(set(image_heights)) > 1:
            raise ValueError(f"Row {index} has images with different heights!")
    # Concatenate images
    max_row_length = max(row_pixel_lengths)
    concatenated_rows = []
    for row in results:
        merged_row = cv2.hconcat(row)
        if merged_row.shape[1] < max_row_length:
            # Add empty image to the end of the row
            merged_row = cv2.hconcat(
                [
                    merged_row,
                    np.ones(
                        (
                            merged_row.shape[0],
                            max_row_length - merged_row.shape[1],
                            merged_row.shape[2],
                        ),
                        dtype=np.uint8,
                    )
                    * 255,
                ]
            )
        concatenated_rows.append(merged_row)
    return cv2.vconcat(concatenated_rows)
