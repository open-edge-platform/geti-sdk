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
import logging
from typing import BinaryIO


class GetiRequestException(Exception):
    """
    Exception representing an unsuccessful http request to the Intel® Geti™ server.
    """

    def __init__(
        self,
        method: str,
        url: str,
        status_code: int,
        request_data: dict[str, dict | str | list | BinaryIO],
        response_data: dict | str | list | None = None,
    ):
        """
        Raise this exception upon unsuccessful requests to the Intel® Geti™ server.

        :param method: Method that was used for the request, e.g. 'POST' or 'GET', etc.
        :param url: URL to which the request was made
        :param status_code: HTTP status code returned for the request
        :param request_data: Data that was included with the request.
        :param response_data: Optional data that was returned in response to the
            request, if any
        """
        self.method = method
        self.url = url
        self.status_code = status_code
        self.request_data = request_data

        self.response_message: str | None = None
        self.response_error_code: str | None = None

        if response_data:
            if isinstance(response_data, dict):
                self.response_message = response_data.get("message", None)
                self.response_error_code = response_data.get("error_code", None)
            elif isinstance(response_data, str):
                self.response_message = response_data
                self.response_error_code = self.status_code
            else:
                logging.warning(
                    f"Received unexpected response type `{type(response_data)}` from server. Response: {response_data}"
                )

    def __str__(self) -> str:
        """
        Return string representation of the unsuccessful http request to the
        Intel® Geti™ server.
        """
        error_str = f"{self.method} request to '{self.url}' failed with status code {self.status_code}."
        if self.response_error_code and self.response_message:
            error_str += (
                f" Server returned error code '{self.response_error_code}' with message '{self.response_message}'"
            )
        return error_str
