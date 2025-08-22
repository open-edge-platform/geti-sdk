# Copyright (C) 2024 Intel Corporation
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

import numpy as np
import requests

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceAction
from geti_sdk.rest_converters import PredictionRESTConverter


class HttpRequestAction(PostInferenceAction):
    """
    Post inference action that will send an http request to a specified `url`.

    :param url: Full URL to send the request to
    :param method: HTTP method to use, for example `GET` or `POST`
    :param headers: Headers to use in the request, can be used for instance for
        authentication purposes
    :param include_prediction_data: Set this to True to include the prediction data in
        the body of the request. Only applicable for POST requests
    :param log_level: Log level for the action. Options are 'info' or 'debug'
    """

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        include_prediction_data: bool = False,
        log_level: str = "info",
    ):
        super().__init__(log_level=log_level)

        self.url = url
        self.method = method

        if headers is None:
            headers = {}
        self.headers = headers

        self.include_data = include_prediction_data and method == "POST"
        if self.include_data:
            self.headers.update({"Content-Type": "application/json"})

        self._repr_info_ = f"url=`{url}`, method={method}, headers={headers}, include_data={self.include_data}"

    def __call__(
        self,
        image: np.ndarray,
        prediction: Prediction,
        score: float | None = None,
        name: str | None = None,
        timestamp: datetime | None = None,
    ):
        """
        Execute the action, send an HTTP request to the predefined url

        :param image: Numpy array representing an image
        :param prediction: Prediction object which was generated for the image
        :param score: Optional score computed from a post inference trigger
        :param name: String containing the name of the image
        :param timestamp: Datetime object containing the timestamp belonging to the
            image
        """
        data = None
        if self.include_data:
            prediction_dict = PredictionRESTConverter.to_dict(prediction)
            data = prediction_dict

        requests.request(method=self.method, url=self.url, headers=self.headers, json=data)
        self.log_function(f"HTTP {self.method} request send to `{self.url}`.")
