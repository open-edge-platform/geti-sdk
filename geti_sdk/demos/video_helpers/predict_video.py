# Copyright (C) 2023 Intel Corporation
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

import logging
import os
import shutil
import subprocess  # nosec B404
import tempfile
import time

import cv2
import imageio_ffmpeg
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.data_models import Prediction
from geti_sdk.deployment import Deployment
from geti_sdk.utils import show_image_with_annotation_scene


def predict_video_from_deployment(
    video_path: str | os.PathLike,
    deployment: Deployment | str | os.PathLike,
    device: str = "CPU",
    preserve_audio: bool | None = True,
) -> str | None:
    """
    Create a video reconstruction with overlaid model predictions.
    This function runs inference on the local machine for every frame in the video.
    The inference results are overlaid on the frames and the output video path will be returned.

    :param video_path: File path to video
    :param deployment: Path to the folder containing the Deployment data, or Deployment instance
    :param device: Device (CPU or GPU) to load the model to. Defaults to 'CPU'
    :param preserve_audio: True to preserve all audio in the original input video. Defaults to True.
        If ffmpeg could not be found, this option is ignored and no audio would be preserved.
    :return: The file path of the output video if generated successfully. Otherwise None.
    """
    retval: str | None = None

    # prepare deployment for running inference
    if isinstance(deployment, str | os.PathLike):
        deployment = Deployment.from_folder(deployment)
    elif not isinstance(deployment, Deployment):
        raise ValueError(f"Unable to read deployment {deployment}")

    logging.info("Load inference models")
    deployment.load_inference_models(device=device)

    # Open the video capture, this prepares the video to be ready for reading
    cap = cv2.VideoCapture(video_path)

    if cap is None or not cap.isOpened():
        raise ValueError(f"Unable to read video from {video_path}")

    # Extract original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_duration = num_frames / fps
    logging.info(f"Input video contains {num_frames:.1f} frames, for a total duration of {video_duration:.1f} seconds")

    t_start = time.time()

    predictions: list[Prediction] = []
    logging.info("Running video prediction... ")
    with (
        logging_redirect_tqdm(tqdm_class=tqdm),
        tqdm(total=num_frames, desc="Predicting") as progress_bar,
    ):
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction = deployment.infer(rgb_frame)
                predictions.append((rgb_frame, prediction))
                progress_bar.update(1)
            else:
                break

    cap.release()

    if len(predictions) == num_frames:
        t_prediction = time.time() - t_start
        logging.info(f"Prediction completed successfully in {t_prediction:.1f} seconds. ")

        # Determine the output video path
        fname, ext = os.path.splitext(video_path)
        output_video_path = os.path.abspath(fname + "_reconstructed" + ext)

        # Create a video writer to be able to save the reconstructed video
        out_video = cv2.VideoWriter(
            filename=output_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_width, frame_height),
        )

        count = 0
        logging.info("Running video reconstruction... ")
        with (
            logging_redirect_tqdm(tqdm_class=tqdm),
            tqdm(total=num_frames, desc="Reconstructing") as progress_bar,
        ):
            for rgb_frame, prediction in predictions:
                output_frame = show_image_with_annotation_scene(
                    image=rgb_frame, annotation_scene=prediction, show_results=False
                )
                out_video.write(output_frame)
                count += 1
                progress_bar.update(1)
        out_video.release()

        if preserve_audio is True:
            try:
                # audio = ffmpeg.input(video_path).audio
                # video = ffmpeg.input(output_video_path).video
                # # ffmpeg can't use same input/output video file.
                # temp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                # out = ffmpeg.output(video, audio, temp.name)
                # out.run(overwrite_output=True, quiet=True)
                # shutil.move(temp.name, output_video_path)
                FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
                logging.info("Restoring all audio in the original input video")
                # ffmpeg can't use same input/output video file.
                with tempfile.NamedTemporaryFile(suffix=ext) as temp:
                    cmd = [
                        FFMPEG,
                        "-i",
                        output_video_path,  # 1st input
                        "-i",
                        video_path,  # 2nd input
                        "-map",
                        "0:v",  # video from the 1st input(output_video_path)
                        "-map",
                        "1:a?",  # audio from the 2nd input(video_path). '?' to ignore if no audio exists.
                        "-c",
                        "copy",
                        "-y",
                        temp.name,
                    ]
                    p = subprocess.run(  # nosec B603
                        cmd,
                        stdin=subprocess.PIPE,
                        capture_output=True,
                    )
                    if p.returncode == 0:
                        shutil.copy(temp.name, output_video_path)
                    else:
                        logging.warning("Error occurred while processing audio. No audio would be preserved.")

            except RuntimeError:  # FFMPEG binary could not be found.
                logging.warning("ffmpeg could not be found on your system. No audio would be preserved.")

        retval = output_video_path
        t_reconstruction = time.time() - t_prediction - t_start
        logging.info(f"Reconstruction completed successfully in {t_reconstruction:.1f} seconds.")
        logging.info(f"Output video saved to `{output_video_path}`")
    else:
        logging.warning("Prediction process failed. ")

    return retval
