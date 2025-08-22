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
import time
import warnings
from datetime import datetime
from functools import cache
from json import JSONDecodeError
from typing import Any

import requests
import simplejson
import urllib3
from requests import Response
from requests.exceptions import RequestException
from requests.structures import CaseInsensitiveDict
from urllib3.exceptions import InsecureRequestWarning

from geti_sdk.platform_versions import GetiVersion

from .exception import GetiRequestException
from .server_config import ServerCredentialConfig, ServerTokenConfig

CSRF_COOKIE_NAME = "_oauth2_proxy_csrf"
PROXY_COOKIE_NAME = "_oauth2_proxy"
GETI_COOKIE_NAME = "geti-cookie"

# INITIAL_HEADERS = {"Connection": "keep-alive", "Upgrade-Insecure-Requests": "1"}
INITIAL_HEADERS = {"Upgrade-Insecure-Requests": "1"}
SUCCESS_STATUS_CODES = [200, 201, 202, 204]

SAAS_MODE = "saas"
ONPREM_MODE = "on-prem"


class GetiSession(requests.Session):
    """
    Wrapper for requests.session that sets the correct headers and cookies, and
    handles authentication and authorization.

    :param server_config: Server configuration holding the hostname (or ip address) of
        the Intel® Geti™ server, as well as the details required for authentication
        (either username and password or personal access token)
    """

    def __init__(
        self,
        server_config: ServerTokenConfig | ServerCredentialConfig,
    ):
        super().__init__()
        self.headers.update(INITIAL_HEADERS)
        self.allow_redirects = False
        self.token = None
        self._cookies: dict[str, str | None] = {
            CSRF_COOKIE_NAME: None,
            PROXY_COOKIE_NAME: None,
        }

        # Configure proxies
        if server_config.proxies is None:
            self._proxies: dict[str, str] = {}
        else:
            self._proxies = server_config.proxies

        # Configure certificate verification
        if not server_config.has_valid_certificate:
            warnings.warn(
                "You have disabled TLS certificate validation, HTTPS requests made to "
                "the Intel® Geti™ server may be compromised. For optimal security, "
                "please enable certificate validation.",
                InsecureRequestWarning,
            )
            urllib3.disable_warnings(InsecureRequestWarning)
        self.verify = server_config.has_valid_certificate

        self.config = server_config
        self.logged_in = False
        self._auth_service: str | None = None

        # Determine authentication method
        if isinstance(server_config, ServerCredentialConfig):
            if self.platform_serving_mode == SAAS_MODE:
                raise ValueError(
                    "Authentication via username and password is not supported for "
                    "Intel® Geti™ SaaS instances. Please use a personal access token."
                )
            logging.warning(
                "Authentication via username and password is deprecated and will be "
                "removed in a future version of the SDK. Please use a personal access token."
            )
            self.authenticate_with_password(verbose=True)
            self.use_token = False
        else:  # ServerTokenConfig
            self.use_token = True
            # New mechanism, from Geti v1.14
            self.headers.update({"x-api-key": f"{server_config.token}"})
            self.headers.pop("Connection")

        # Get server version
        self._product_info = self._get_product_info_and_set_api_version()
        self._organization_id: str | None = self._get_organization_id()

    @property
    @cache
    def platform_serving_mode(self) -> str:
        """
        Return the type of the GETi platform service.
        """
        try:
            deployment_config_response = self.request(
                url=f"{self.config.host}/deployment-config.json",
                method="GET",
                proxies=self._proxies,
            ).json()
            serving_mode = deployment_config_response.get("servingMode").lower()
        except requests.exceptions.JSONDecodeError:
            return ONPREM_MODE
        if serving_mode == "on-prem":
            return ONPREM_MODE
        if serving_mode == "saas":
            return SAAS_MODE
        raise ValueError(f"Unexpected serving mode '{serving_mode}' received from the server.")

    @property
    def version(self) -> GetiVersion:
        """
        Return the version of the Intel® Geti™ platform that is running on the server.

        :return: Version object holding the Intel® Geti™ version number
        """
        if "build-version" in self._product_info:
            version_string = self._product_info.get("build-version")
        else:
            version_string = self._product_info.get("product-version")
        return GetiVersion(version_string)

    @property
    def organization_id(self) -> str:
        """
        Return the organization id for the user who is associated with the GetiSession
        """
        if self._organization_id is None:
            self._organization_id = self._get_organization_id()
        return self._organization_id

    def _get_initial_login_url(self) -> str:
        """
        Retrieve the initial login url by making a request to the login page, and
        following the redirects.

        :return: string containing the URL to the login page
        """
        params = (
            "client_id=web_ui"
            "&redirect_uri=/callback"
            "&code_challenge_method=S256"
            "&response_type=code"
            "&response_mode=query"
            "&scope=openid+profile+groups+email+offline_access"
        )
        url = f"{self.config.host}/dex/auth/regular_users?{params}"
        response = self.get(url, allow_redirects=False, proxies=self._proxies)
        return response.next.url if response.status_code in [302, 303] else response.url

    def authenticate_with_password(self, verbose: bool = True):
        """
        Get a new authentication cookie from the server.

        :param verbose: True to print progress output, False to suppress output
        """
        if self.logged_in:
            logging.info("Already logged in, authentication is skipped")
            return
        self.cookies.clear()
        try:
            login_path = self._get_initial_login_url()
        except requests.exceptions.SSLError as error:
            raise requests.exceptions.SSLError(
                f"Connection to Intel® Geti™ server at '{self.config.host}' failed, "
                f"the server address can be resolved but the SSL certificate could not "
                f"be verified. \n Full error description: {error.args[-1]}"
            )
        except requests.exceptions.ConnectionError as error:
            if "dummy" in self.config.password or "dummy" in self.config.username:
                raise ValueError(
                    "Connection to the Intel® Geti™ server failed, please make sure to "
                    "update the user login information for the Intel® Geti™ instance."
                ) from error
            raise ValueError(
                f"Connection to the Intel® Geti™ server at host '{self.config.host}' "
                f"failed, please provide a valid cluster hostname or ip address as"
                f" well as valid login details."
            ) from error
        self.headers.clear()
        self.headers.update({"Content-Type": "application/x-www-form-urlencoded"})
        if verbose:
            logging.info(f"Authenticating on host {self.config.host}...")
        cookies = {}
        headers = self.headers
        response = self.post(
            url=login_path,
            data={"login": self.config.username, "password": self.config.password},
            cookies=cookies,
            headers=headers,
            allow_redirects=True,
            proxies=self._proxies,
        )
        self._handle_dex_response(response)
        if verbose:
            logging.info("Authentication successful. Cookie received.")
        self.logged_in = True

    def get_rest_response(
        self,
        url: str,
        method: str,
        contenttype: str = "json",
        data: Any | None = None,
        allow_reauthentication: bool = True,
        include_organization_id: bool = True,
        allow_text_response: bool = False,
        request_headers: dict[str, str] = {},
    ) -> Response | dict | list:
        """
        Return the REST response from a request to `url` with `method`.

        :param url: the REST url without the hostname and api pattern
        :param method: 'GET', 'POST', 'PUT', 'DELETE'
        :param contenttype: currently either 'json', 'jpeg', 'multipart', 'zip', or '',
            defaults to "json"
        :param data: the data to send in a post request, as json
        :param allow_reauthentication: True to handle authentication errors
            by attempting to re-authenticate. If set to False, such errors
            will be raised instead.
        :param include_organization_id: True to include the organization ID in the base
            URL. Can be set to False for accessing certain internal endpoints that do
            not require an organization ID, but do require error handling.
        :param allow_text_response: False to trigger error handling when the server
            returns a response with text/html content. This can happen in some cases
            when authentication has expired. However, some endpoints are designed to
            return text responses, for those endpoints this parameter should be set to
            True
        :param request_headers: Additional headers to include in the request
        """
        if self.config.api_pattern in url:
            url = url.split(self.config.api_pattern)[-1]

        self._update_headers_for_content_type(content_type=contenttype)

        if not include_organization_id or url.startswith(f"organizations/{self.organization_id}/"):
            requesturl = f"{self.config.base_url}{url}"
        else:
            requesturl = f"{self.base_url}{url}"

        if method == "POST" or method == "PUT" or method == "PATCH":
            if contenttype == "json":
                kw_data_arg = {"json": data}
            elif contenttype == "multipart":
                kw_data_arg = {"files": data}
            elif contenttype == "jpeg" or contenttype == "zip" or contenttype == "offset+octet-stream":
                kw_data_arg = {"data": data}
            else:
                raise ValueError(
                    f"Making a POST request with content of type {contenttype} is "
                    f"currently not supported through the Intel Geti SDK."
                )
        else:
            kw_data_arg = {}

        request_params = {
            "method": method,
            "url": requesturl,
            **kw_data_arg,
            "stream": True,
        }

        if not self.use_token:
            request_params.update({"cookies": self._cookies})

        csrf_header_methods = ["POST", "PUT", "DELETE", "PATCH"]
        if method in csrf_header_methods:
            self.headers.update({"x-geti-csrf-protection": "1"})
        else:
            self.headers.pop("x-geti-csrf-protection", "")

        # Make the request, retrying a maximum of 5 times in case of connection errors
        retries = 5
        last_conn_error: ConnectionError | None = None
        while retries:
            try:
                response = self.request(**request_params, proxies=self._proxies, headers=request_headers)
                break
            except requests.exceptions.SSLError as error:
                raise requests.exceptions.SSLError(
                    f"Connection to Intel® Geti™ server at '{self.config.host}' failed, "
                    f"the server address can be resolved but the SSL certificate could not "
                    f"be verified. \n Full error description: {error.args[-1]}"
                )
            except ConnectionError as conn_error:
                last_conn_error = conn_error
                retries -= 1

        if last_conn_error is not None:
            raise last_conn_error
        response_content_type = response.headers.get("Content-Type", [])
        if response.status_code not in SUCCESS_STATUS_CODES or (
            "text/html" in response_content_type and not allow_text_response
        ):
            response = self._handle_error_response(
                response=response,
                request_params=request_params,
                request_data=kw_data_arg,
                allow_reauthentication=allow_reauthentication,
                content_type=contenttype,
            )
        if response.headers.get("Content-Type", "").startswith("application/json") and response.status_code != 204:
            result = response.json()
        else:
            result = response

        return result

    def logout(self, verbose: bool = True) -> None:
        """
        Log out of the server and end the session. All HTTPAdapters are closed and
        cookies and headers are cleared.

        :param verbose: False to suppress output messages
        """
        requires_closing = True
        requires_sign_out = True

        if not self.logged_in or self.use_token:
            requires_closing = False
            requires_sign_out = False

        if requires_sign_out:
            sign_out_url = self.config.base_url[: -len(self.config.api_pattern)] + "/oauth2/sign_out"
            try:
                response = self.request(url=sign_out_url, method="GET", proxies=self._proxies)

                if response.status_code in SUCCESS_STATUS_CODES:
                    if verbose:
                        logging.info("Logout successful.")
                else:
                    raise GetiRequestException(
                        method="GET",
                        url=sign_out_url,
                        status_code=response.status_code,
                        request_data={},
                    )
            except (RequestException, AttributeError):
                if verbose:
                    logging.info(
                        f"The {self.__class__.__name__} is closed successfully, but "
                        f"the Geti instance was not able to logout from the server."
                    )
            except TypeError:
                # The session is already closed
                requires_closing = False

        self._cookies = {}
        self.cookies.clear()
        self.headers = CaseInsensitiveDict(INITIAL_HEADERS)
        self.logged_in = False
        if requires_closing:
            try:
                super().close()
            except TypeError as error:
                # Sometimes a TypeError is raised if logout is called during garbage
                # collection, however we can safely ignore this since the session will
                # be deleted in that case anyway.
                logging.debug(f"{error} encountered during GetiSession closure.")

    def _get_product_info_and_set_api_version(self) -> dict[str, str]:
        """
        Return the product info as retrieved from the Intel® Geti™ server.

        This method will also attempt to set the API version correctly, based on the
        retrieved product info.

        :return: Dictionary containing the product info.
        """
        return self.get_rest_response("product_info", "GET", include_organization_id=False)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Log out of the Intel® Geti™ server. This method is called when exiting the
        runtime context related to the session, in case the session is used as a context
        manager.
        """
        if self.logged_in:
            try:
                self.logout(verbose=False)
            except Exception as exc:
                logging.debug(f"{exc} encountered while exiting GetiSession context.")
        super().__exit__(exc_type, exc_value, traceback)

    def __del__(self):
        """
        Log out of the Intel® Geti™ server. This method is called when the session is
        deleted from memory.
        """
        if self.logged_in:
            try:
                self.logout(verbose=False)
            except Exception as exc:
                logging.debug(f"{exc} encountered during GetiSession deletion.")

    def _handle_error_response(
        self,
        response: Response,
        request_params: dict[str, Any],
        request_data: dict[str, Any],
        allow_reauthentication: bool = True,
        content_type: str = "json",
    ) -> Response:
        """
        Handle error responses from the server.

        :param response: The Response object received from the server
        :param request_params: Dictionary containing the original parameters of the
            request
        :param allow_reauthentication: True to handle authentication errors
            by attempting to re-authenticate. If set to False, such errors
            will be raised instead.
        :param content_type: The content type of the original request
        :raises: GetiRequestException in case the error cannot be handled
        :return: Response object resulting from the request
        """
        retry_request = False

        if response.status_code in [200, 401] and allow_reauthentication:
            # Authentication has likely expired, re-authenticate
            logging.info("Authentication may have expired, re-authenticating...")
            self.logged_in = False
            if not self.use_token:
                self.authenticate_with_password(verbose=False)
                logging.info("Authentication complete.")

            else:
                raise ValueError(
                    "Authentication via your personal access token has failed, "
                    "most likely the token has expired. Please verify that you "
                    "have provided a valid token."
                )

            retry_request = True

        elif response.status_code == 503:
            # In case of Service Unavailable, wait some time and try again. If it
            # still doesn't work, raise exception
            time.sleep(1)
            retry_request = True

        # We make one attempt to do the request again. If it fails again, a
        # GetiRequestException will be raised holding further details of the
        # reason for failure.
        if retry_request:
            self._update_headers_for_content_type(content_type=content_type)
            # Reset any file buffers that were included in the request data, so that we
            # can attempt to upload them again.
            if content_type == "multipart":
                for file_name, file_buffer in request_params["files"].items():
                    file_buffer.seek(0, 0)

            response = self.request(**request_params, proxies=self._proxies)

            if response.status_code in SUCCESS_STATUS_CODES:
                return response

        try:
            response_data = response.json()
        except (JSONDecodeError, simplejson.errors.JSONDecodeError):
            response_data = None

        raise GetiRequestException(
            method=request_params["method"],
            url=request_params["url"],
            status_code=response.status_code,
            request_data=request_data,
            response_data=response_data,
        )

    def _update_headers_for_content_type(self, content_type: str) -> None:
        """
        Update the session headers to contain the correct content type

        :param content_type: content type for the request
        """
        if content_type == "json":
            self.headers.update({"Content-Type": "application/json"})
        elif content_type == "jpeg":
            self.headers.update({"Content-Type": "image/jpeg"})
        elif content_type == "multipart" or content_type == "":
            self.headers.pop("Content-Type", None)
        elif content_type == "zip":
            self.headers.update({"Content-Type": "application/zip"})
        elif content_type == "offset+octet-stream":
            self.headers.update({"Content-Type": "application/offset+octet-stream"})
        else:
            self.headers.update({"Content-Type": content_type})

    @property
    def base_url(self) -> str:
        """
        Return the base URL to the Intel Geti server. If the server is running
        Geti v1.9 or later, the organization ID will be included in the URL
        """
        return f"{self.config.base_url}organizations/{self.organization_id}/"

    def _get_organization_id(self) -> str:
        """
        Return the organization ID associated with the user and host information configured
        in this Session.

        NOTE: When authenticating with username and password, this method returns the
        ID of the default organization for the user!
        """
        if not self.use_token:
            result = self.get_rest_response(
                url="profile",
                method="GET",
                include_organization_id=False,
            )
        else:
            result = self.get_rest_response(
                url="personal_access_tokens/organization",
                method="GET",
                include_organization_id=False,
            )
        if "organizationId" in result:
            # Geti < 2.5, return the id directly
            org_id = result.get("organizationId", None)
        elif "organizations" in result:
            # Geti 2.5 and up: list of organizations, return the default one (the one
            # which was created at the earliest time)
            org_list = result["organizations"]
            creation_times = [
                datetime.fromisoformat(x["organizationCreatedAt"].replace("Z", "+00:00")) for x in org_list
            ]
            ids = [x["organizationId"] for x in org_list]
            earliest_creation_time = min(creation_times)
            earliest_idx = creation_times.index(earliest_creation_time)
            org_id = ids[earliest_idx]
        else:
            org_id = None

        if org_id is None:
            raise ValueError(
                f"Unable to retrieve organization ID from the Intel Geti server. Server responded with: `{result}`"
            )
        return org_id

    def _handle_legacy_dex_response(self, response: Response) -> None:
        """
        Process the login response for the old-style DEX authorization mechanism

        :param response: Response object to handle
        """
        try:
            previous_response = response.history[-1]
        except IndexError:
            raise ValueError(
                "The cluster responded to the request, but authentication failed. "
                "Please verify that you have provided correct credentials."
            )
        proxy_cookie = previous_response.cookies.get(PROXY_COOKIE_NAME)
        if proxy_cookie is not None:
            cookie = {PROXY_COOKIE_NAME: proxy_cookie}
            self._cookies.update(cookie)
        else:
            logging.warning(
                f"Authentication appears to have failed! No valid oauth cookie "
                f"obtained. Invalid response received from server. Status code: "
                f"{response.status_code}"
            )

    def _handle_dex_response(self, response: Response) -> None:
        """
        Process the login response from the DEX authorization mechanism

        Extract the code and exchange it for a token, which must then
        be used as authentication cookie

        :param response: DEX response object to handle
        """
        if response.status_code == 200:
            code = response.url.split("code=")[1].split("&")[0]
        elif response.status_code == 401:
            raise ValueError(
                "The cluster responded to the request, but authentication failed. "
                "Please verify that you have provided correct credentials."
            )
        elif response.status_code == 404:
            raise ValueError(
                "Unable to authenticate with the Intel Geti server. The authentication "
                "mechanism you are trying to use is no longer supported. This error "
                "indicates that the Intel® Geti™ server version is not supported by "
                "this version of the Intel Geti SDK package. Please update the "
                "Intel® Geti™ server to version 2.0 or later, or use a previous "
                "version of the SDK."
            )
        else:
            raise GetiRequestException(
                method=response.request.method,
                url=response.url,
                status_code=response.status_code,
                request_data={},
                response_data={
                    "message": response.content,
                    "error_code": response.reason,
                },
            )
        data = {
            "grant_type": "authorization_code",
            "redirect_uri": "/callback",
            "code": code,
            "client_id": "web_ui",
        }
        login_response = self.post(
            url=f"{self.config.host}/dex/token",
            data=data,
            allow_redirects=True,
            proxies=self._proxies,
        )
        token = login_response.json().get("access_token", None)
        self._cookies.update({GETI_COOKIE_NAME: token})
