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

import base64
import io
import json
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.data_models import (
    AnnotationKind,
    Dataset,
    Image,
    MediaItem,
    Prediction,
    Project,
    Video,
    VideoFrame,
)
from geti_sdk.data_models.containers import MediaList
from geti_sdk.data_models.enums import MediaType, PredictionMode
from geti_sdk.data_models.predictions import ResultMedium
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_converters.prediction_rest_converter import PredictionRESTConverter


class PredictionClient:
    """
    Class to download predictions from an existing Intel® Geti™ project.
    """

    def __init__(self, session: GetiSession, project: Project, workspace_id: str):
        self.session = session
        self.project = project
        self._base_url = f"workspaces/{workspace_id}/projects/{project.id}/"
        self._labels = project.get_all_labels()
        self.__project_ready = self.__are_models_trained()
        self._mode = PredictionMode.AUTO
        self.__override_mode: PredictionMode | None = None

    def __are_models_trained(self) -> bool:
        """
        Check that the project to which this PredictionClient belongs has trained
        models to generate predictions from. This method will return True if at least
        one model is trained for each task in the project task chain.

        :return: True if the project has trained models and is ready to generate
            predictions, False otherwise
        """
        response = self.session.get_rest_response(url=f"{self._base_url}model_groups", method="GET")
        model_info_array: list[dict[str, Any]]

        model_info_array = response.get("model_groups", [])

        task_ids = [task.id for task in self.project.get_trainable_tasks()]
        tasks_with_models: list[str] = []
        for item in model_info_array:
            if len(item["models"]) > 0:
                tasks_with_models.append(item["task_id"])

        # It is sufficient if the FIRST task has a model trained
        return task_ids[0] in tasks_with_models

    @property
    def ready_to_predict(self):
        """
        Return True if the project is ready to yield predictions, False otherwise.

        :return:
        """
        if self.__project_ready:
            return True
        self.__project_ready = self.__are_models_trained()
        return self.__project_ready

    @property
    def mode(self) -> PredictionMode:
        """
        Return the current mode used to retrieve predictions. There are three options:
         - auto
         - latest
         - online

        Auto will fetch prediction from the database if it is up to date and
        otherwise send an inference request. Online will always send an
        inference request and latest will not send an inference request but grabs
        the latest result from the database.

        By default, the mode is set to `auto`.

        :return: Current PredictionMode used to retrieve predictions
        """
        if self.__override_mode is None:
            return self._mode
        return self.__override_mode

    @mode.setter
    def mode(self, new_mode: str | PredictionMode):
        """
        Set the mode for the Prediction client to retrieve predictions from the
        Intel® Geti™ server.

        :param new_mode: PredictionMode (or string representing a prediction mode) to
            set
        """
        if isinstance(new_mode, str):
            new_mode = PredictionMode(new_mode)
        self._mode = new_mode

    def _get_prediction_for_media_item(
        self,
        media_item: MediaItem,
        dataset: Dataset | None = None,
        prediction_mode: PredictionMode = PredictionMode.AUTO,
        include_explanation: bool = False,
    ) -> tuple[Prediction | list[Prediction] | None, str]:
        """
        Get the prediction for a media item. If a 2D media item (Image or VideoFrame)
        is passed, this method will return a single Prediction. If a Video is passed,
        this method will return a list of predictions.

        In case of failure to get a prediction, the first element of the tuple
        returned by this method will be None, and the second will be a message
        describing the problem.

        :param media_item: Image, Video or VideoFrame to get the prediction for
        :param dataset: Dataset to which the media item belongs. Only needs to be
            specified when the project contains more than one dataset
        :param prediction_mode: Mode for the predictions to be retrieved. Use
            ONLINE to always generate a new prediction, or LATEST to obtain the latest
            prediction from the cache. AUTO can be used to first query the cache for
            a prediction, and if none is found the system will generate a new one on
            the fly
        :param include_explanation: Only applicable to Geti v1.13 or higher: True
            to include saliency maps in the result. If set to False, saliency maps
            will be omitted. Setting this to True will increase the time required to
            retrieve the prediction
        :return: Tuple containing:
         - Prediction (for Image/VideoFrame) or List of Predictions (for Video)
         - string containing a message
        """
        if not self.ready_to_predict:
            msg = (
                f"Not all tasks in project '{self.project.name}' have a trained "
                f"model available. Unable to get predictions from the project."
            )
            return None, msg
        if dataset is None:
            if len(self.project.datasets) == 1:
                dataset = self.project.training_dataset
                dataset_id = dataset.id
            else:
                logging.debug(
                    f"The project {self.project.name} contains multiple datasets, but "
                    f"the dataset for the media item was not specified. The "
                    f"PredictionClient will try to compute the correct dataset for the "
                    f"media item, but this may lead to a 404 `resource_not_found` error. "
                    f"For optimal performance, please make sure to specify the correct "
                    f"dataset"
                )
                dataset_id_string = media_item.base_url.split("/datasets/")[-1]
                dataset_id = dataset_id_string.split("/")[0]
        else:
            dataset_id = dataset.id
        data: dict[str, str] | None = None
        explain_response: dict | None = None

        prediction_mode_map = {
            "auto": "auto",
            "online": "never",
            "latest": "always",
        }
        if media_item.type == MediaType.IMAGE:
            data = {
                "image_id": media_item.identifier.image_id,
                "dataset_id": dataset_id,
            }
            action = "predict"
        elif media_item.type == MediaType.VIDEO_FRAME:
            data = {
                "dataset_id": dataset_id,
                "video_id": media_item.identifier.video_id,
                "frame_index": str(media_item.identifier.frame_index),
            }
            action = "predict"
        elif media_item.type == MediaType.VIDEO:
            stride = media_item.media_information.frame_stride
            data = {
                "dataset_id": dataset_id,
                "video_id": media_item.identifier.video_id,
                "start_frame": 0,
                "end_frame": int(15 * stride),
                "frame_skip": stride,
            }
            action = "batch_predict"
        url = f"{self._base_url}pipelines/active:{action}?use_cache={prediction_mode_map[str(prediction_mode)]}"
        include_org_id = True
        method = "POST"
        if include_explanation:
            explain_map = {"predict": "explain", "batch_predict": "batch_explain"}
            explain_url = f"{self._base_url}pipelines/active:{explain_map[action]}"
            explain_response = self.session.get_rest_response(
                url=explain_url,
                method="POST",
                include_organization_id=True,
                data=data,
            )
        try:
            response = self.session.get_rest_response(
                url=url,
                method=method,
                include_organization_id=include_org_id,
                data=data,
            )
            if isinstance(media_item, Image | VideoFrame):
                result = PredictionRESTConverter.from_dict(response)
                if include_explanation:
                    maps: list[ResultMedium] = []
                    for map_dict in explain_response.get("maps", []):
                        saliency_map = ResultMedium(name="saliency_map", label_id=map_dict.get("label_id", None))
                        saliency_map.data = base64.b64decode(map_dict["data"])
                        maps.append(saliency_map)
                    result.maps = maps
                result.resolve_labels_for_result_media(labels=self._labels)
                result.resolve_label_names_and_colors(labels=self._labels)

            elif isinstance(media_item, Video):
                result = []
                for ind, prediction in enumerate(response["video_predictions"]):
                    pred_object = PredictionRESTConverter.from_dict(prediction)
                    pred_object.resolve_label_names_and_colors(labels=self._labels)
                    if include_explanation:
                        maps: list[ResultMedium] = []
                        for map_dict in explain_response["explanations"][ind].get("maps", []):
                            saliency_map = ResultMedium(
                                name="saliency_map",
                                label_id=map_dict.get("label_id", None),
                            )
                            saliency_map.data = map_dict["data"].encode("utf-8")
                            maps.append(saliency_map)
                        pred_object.maps = maps
                    pred_object.resolve_labels_for_result_media(labels=self._labels)
                    result.append(pred_object)
            else:
                raise TypeError(
                    f"Getting predictions is not supported for media item of type "
                    f"{media_item.type}. Unable to retrieve predictions."
                )
            msg = "success"
        except GetiRequestException as error:
            msg = f"Unable to retrieve prediction for {media_item.type}."
            if error.status_code == 204:
                msg += (
                    f" The prediction for the {media_item.type} with name "
                    f"'{media_item.name}' is not available in project "
                    f"'{self.project.name}'."
                )
                if prediction_mode == PredictionMode.LATEST:
                    msg += (
                        "Try setting the mode of the prediction client to "
                        "'auto' or 'online' to trigger inference upon request."
                    )
            else:
                msg += f" Server responded with error message: {str(error)}"
            result = None
        return result, msg

    def get_image_prediction(self, image: Image) -> Prediction:
        """
        Get a prediction for an image from the Intel® Geti™ server, if available.

        NOTE: This method is only available for images that are already existing on
        the server! For getting predictions on a 'new' image, please see the
        `PredictionClient.predict_image` method

        :param image: Image to get the prediction for. The image has to be present in
            the project on the cluster already.
        :return: Prediction for the image
        """
        result, msg = self._get_prediction_for_media_item(media_item=image, prediction_mode=self.mode)
        if result is None:
            raise ValueError(msg)
        return result

    def get_video_frame_prediction(self, video_frame: VideoFrame) -> Prediction:
        """
        Get a prediction for a video frame from the Intel® Geti™ server, if available.

        :param video_frame: VideoFrame to get the prediction for. The frame has to be
            present in the project on the cluster already.
        :return: Prediction for the video frame
        """
        result, msg = self._get_prediction_for_media_item(media_item=video_frame, prediction_mode=self.mode)
        if result is None:
            raise ValueError(msg)
        return result

    def get_video_predictions(self, video: Video) -> list[Prediction]:
        """
        Get a list of predictions for a video from the Intel® Geti™ server, if available.

        :param video: Video to get the predictions for. The video has to be present in
            the project on the cluster already.
        :return: List of Predictions for the video
        """
        result, msg = self._get_prediction_for_media_item(media_item=video, prediction_mode=self.mode)
        if result is None:
            raise ValueError(msg)
        return result

    def download_predictions_for_images(
        self,
        images: MediaList[Image],
        path_to_folder: str,
        include_result_media: bool = True,
    ) -> float:
        """
        Download image predictions from the server to a target folder on disk.

        :param images: List of images for which to download the predictions
        :param path_to_folder: Folder to save the predictions to
        :param include_result_media: True to also download the result media belonging
            to the predictions, if any. False to skip downloading result media
        :return: Returns the time elapsed to download the predictions, in seconds
        """
        return self._download_predictions_for_2d_media_list(
            media_list=images,
            path_to_folder=path_to_folder,
            include_result_media=include_result_media,
        )

    def download_predictions_for_videos(
        self,
        videos: MediaList[Video],
        path_to_folder: str,
        include_result_media: bool = True,
        inferred_frames_only: bool = True,
        frame_stride: int | None = None,
    ) -> float:
        """
        Download predictions for a list of videos from the server to a target folder
        on disk.

        :param videos: List of videos for which to download the predictions
        :param path_to_folder: Folder to save the predictions to
        :param include_result_media: True to also download the result media belonging
            to the predictions, if any. False to skip downloading result media
        :param inferred_frames_only: True to only download frames that already have
            a prediction, False to run inference on the full video for all videos in
            the list.
            WARNING: Setting this to False may cause the download to take a long time!
        :param frame_stride: Optional frame stride to use when generating predictions.
            This is only used when `inferred_frames_only = False`. If left unspecified,
            the frame_stride is deduced from the video
        :return: Time elapsed to download the predictions, in seconds
        """
        t_total = 0
        logging.info(
            f"Starting prediction download... saving predictions for "
            f"{len(videos)} videos to folder {path_to_folder}/predictions"
        )
        for video in videos:
            t_total += self.download_predictions_for_video(
                video=video,
                path_to_folder=path_to_folder,
                include_result_media=include_result_media,
                inferred_frames_only=inferred_frames_only,
                frame_stride=frame_stride,
            )
        logging.info(f"Video prediction download finished in {t_total:.1f} seconds.")
        return t_total

    def download_predictions_for_video(
        self,
        video: Video,
        path_to_folder: str,
        include_result_media: bool = True,
        inferred_frames_only: bool = True,
        frame_stride: int | None = None,
    ) -> float:
        """
        Download video predictions from the server to a target folder on disk.

        :param video: Video for which to download the predictions
        :param path_to_folder: Folder to save the predictions to
        :param include_result_media: True to also download the result media belonging
            to the predictions, if any. False to skip downloading result media
        :param inferred_frames_only: True to only download frames that already have
            a prediction, False to run inference on the full video.
            WARNING: Setting this to False may cause the download to take a long time!
        :param frame_stride: Optional frame stride to use when generating predictions.
            This is only used when `inferred_frames_only = False`. If left unspecified,
            the frame_stride is deduced from the video
        :return: Returns the time elapsed to download the predictions, in seconds
        """
        if inferred_frames_only:
            predictions = self.get_video_predictions(video=video)
            frame_list = MediaList[VideoFrame](
                [
                    VideoFrame.from_video(video=video, frame_index=prediction.media_identifier.frame_index)
                    for prediction in predictions
                ]
            )
        else:
            stride = (
                frame_stride if frame_stride is not None and frame_stride > 0 else video.media_information.frame_stride
            )
            frame_indices = range(0, video.media_information.frame_count, stride)
            frame_list = MediaList[VideoFrame](
                [VideoFrame.from_video(video=video, frame_index=frame_index) for frame_index in frame_indices]
            )
            # Set the prediction mode to online to force inference on frames that don't
            # have a prediction yet
            self._override_mode(PredictionMode.ONLINE)
        if len(frame_list) > 0:
            result = self._download_predictions_for_2d_media_list(
                media_list=frame_list,
                path_to_folder=path_to_folder,
                verbose=False,
                include_result_media=include_result_media,
            )
        else:
            result = 0
        self._reset_override_mode()
        return result

    def _override_mode(self, mode: PredictionMode):
        """
        Temporarily override the prediction mode.

        :param mode: new prediction mode to use
        :return:
        """
        self.__override_mode = mode

    def _reset_override_mode(self):
        """
        Remove override of the prediction mode.

        :return:
        """
        self.__override_mode = None

    def _download_predictions_for_2d_media_list(
        self,
        media_list: MediaList[Image] | MediaList[VideoFrame],
        path_to_folder: str,
        include_result_media: bool = True,
        verbose: bool = True,
    ) -> float:
        """
        Download predictions from the server to a target folder on disk.

        :param media_list: List of images or video frames to download the predictions
            for
        :param path_to_folder: Folder to save the predictions to
        :param include_result_media: True to also download the result media belonging
            to the predictions, if any. False to skip downloading result media
        :param verbose: True to print verbose output, False to run in silent mode
        :return: Returns the time elapsed to download the predictions, in seconds
        """
        if media_list.media_type == Image:
            media_name = "image"
            media_name_plural = "images"
        elif media_list.media_type == VideoFrame:
            media_name = "video frame"
            media_name_plural = "video frames"
        else:
            raise ValueError("Invalid media type found in media_list, unable to download predictions.")

        if not path_to_folder.endswith("predictions"):
            path_to_predictions_folder = os.path.join(path_to_folder, "predictions")
        else:
            path_to_predictions_folder = path_to_folder

        if verbose:
            logging.info(
                f"Starting prediction download... saving predictions for "
                f"{len(media_list)} {media_name_plural} to folder "
                f"{path_to_predictions_folder}"
            )
        os.makedirs(path_to_predictions_folder, exist_ok=True, mode=0o770)
        t_start = time.time()
        download_count = 0
        skip_count = 0
        tqdm_prefix = "Downloading predictions"
        with logging_redirect_tqdm(tqdm_class=tqdm):
            for media_item in tqdm(media_list, desc=tqdm_prefix):
                prediction, msg = self._get_prediction_for_media_item(
                    media_item,
                    prediction_mode=self.mode,
                    include_explanation=include_result_media,
                )
                if prediction is None:
                    if verbose:
                        logging.info(
                            f"Unable to retrieve prediction for {media_name} "
                            f"{media_item.name}, with reason: {msg}. Skipping this "
                            f"{media_name}"
                        )
                    skip_count += 1
                    continue
                kind = prediction.kind
                if kind != AnnotationKind.PREDICTION:
                    if verbose:
                        logging.warning(
                            f"Received invalid prediction of kind {kind} for {media_name} with name{media_item.name}"
                        )
                    skip_count += 1
                    continue

                # Download result media belonging to the prediction, if required
                if prediction.has_result_media and include_result_media:
                    try:
                        result_media = prediction.get_result_media_data(self.session)
                    except GetiRequestException:
                        if verbose:
                            logging.info(
                                f"Unable to retrieve prediction result map for "
                                f"{media_name} '{media_item.name}'. Skipping"
                            )
                        result_media = None
                    if result_media is not None:
                        path_to_result_media_folder = os.path.join(path_to_predictions_folder, "saliency_maps")
                        os.makedirs(path_to_result_media_folder, exist_ok=True, mode=0o770)
                        for result_medium in result_media:
                            result_media_path = os.path.join(
                                path_to_result_media_folder,
                                media_item.name + "_" + result_medium.friendly_name + ".jpg",
                            )

                            os.makedirs(
                                os.path.dirname(result_media_path),
                                exist_ok=True,
                                mode=0o770,
                            )
                            with open(result_media_path, "wb") as f:
                                f.write(result_medium.data)

                # Convert prediction to json and save to file
                export_data = PredictionRESTConverter.to_dict(prediction)
                prediction_path = os.path.join(path_to_predictions_folder, media_item.name + ".json")

                os.makedirs(os.path.dirname(prediction_path), exist_ok=True, mode=0o770)
                with open(prediction_path, "w") as f:
                    json.dump(export_data, f, indent=4)
                download_count += 1
        t_elapsed = time.time() - t_start
        if download_count > 0:
            msg = (
                f"Downloaded {download_count} predictions to folder "
                f"{path_to_predictions_folder} in {t_elapsed:.1f} seconds."
            )
        else:
            msg = "No predictions were downloaded."
        if skip_count > 0:
            msg = (
                msg + f" Was unable to retrieve predictions for {skip_count} "
                f"{media_name_plural}, these {media_name_plural} were skipped."
            )
        if verbose:
            logging.info(msg)
        return t_elapsed

    def predict_image(self, image: Image | np.ndarray | os.PathLike | str) -> Prediction:
        """
        Push an image to the Intel® Geti™ project and receive a prediction for it.

        Note that this method will not save the image to the project.

        :param image: Image object, filepath to an image or numpy array containing an
            image to get the prediction for
        :return: Prediction for the image
        """
        # Get image pixel data from input
        image_data: np.ndarray | None
        image_name: str | None
        if isinstance(image, Image):
            image_data = image.numpy
            if image_data is None:
                raise ValueError(
                    "An 'Image' object was passed for prediction, but the image does "
                    "not contain any pixel data. Please make sure that the pixel data "
                    "is loaded for the image by calling 'image.get_data()' with the "
                    "appropriate Geti session."
                )
            image_name = image.name
        elif isinstance(image, np.ndarray):
            image_data = image
            image_name = "numpy_image.jpg"
        elif isinstance(image, os.PathLike | str):
            image_data = None
            image_name = None
        else:
            raise TypeError(
                f"Received object 'image' of type {type(image)}, which is invalid. "
                f"Please either pass an 'Image' object, a numpy array or a filepath."
            )

        if image_data is None:
            image_io = open(image, "rb").read()  # noqa: SIM115
        else:
            image_io = io.BytesIO(cv2.imencode(".jpg", image_data)[1].tobytes())
            image_io.name = image_name

        url = f"{self._base_url}pipelines/active:predict"
        contenttype = "multipart"
        data = {"file": image_io}

        # make POST request
        response = self.session.get_rest_response(
            url=url,
            method="POST",
            contenttype=contenttype,
            data=data,
        )
        prediction = PredictionRESTConverter.from_dict(response)
        prediction.resolve_label_names_and_colors(labels=self._labels)
        return prediction
