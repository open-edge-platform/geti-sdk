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


import attrs

DEFAULT_API_VERSION = "v1"
LEGACY_API_VERSION = "v1.0"


def trim_trailing_slash(input_string: str) -> str:
    """
    Remove trailing slash(es) from a string, if present.

    :param input_string: String to remove the trailing slash(es) from
    :return: Updated string
    """
    return input_string.rstrip("/")


@attrs.define(slots=False)
class ServerConfig:
    """
    Base configuration holding the connection details of the Intel® Geti™ server.
    Contains the hostname, ssl certificate configuration and proxy configuration.

    :var host: full hostname or ip address of the Intel® Geti™ server.
        Note: this should include the protocol (i.e. https://your_geti_hostname.com)
    :var has_valid_certificate: Set to True if the server has a valid SSL certificate
        that should be validated and used to establish an encrypted HTTPS connection
    :var proxies: Optional dictionary containing proxy information, if this is
        required to connect to the server. For example:

        proxies = {
            'http': http://proxy-server.com:<http_port_number>,
            'https': http://proxy-server.com:<https_port_number>
        },

        if set to None (the default), the global proxy settings found on the system
        will be used. If set to an emtpy dictionary, no proxy will be used.
    """

    host: str = attrs.field(converter=trim_trailing_slash)
    has_valid_certificate: bool = attrs.field(default=False, kw_only=True)
    proxies: dict[str, str] | None = attrs.field(default=None, kw_only=True)

    def __attrs_post_init__(self):
        """
        Initialize private attributes
        """
        self._api_version = DEFAULT_API_VERSION

        # Sanitize hostname
        if not self.host.startswith("https://"):
            if self.host.startswith("http://"):
                raise ValueError("HTTP connections are not supported, please use HTTPS instead.")
            self.host = "https://" + self.host

    @property
    def base_url(self) -> str:
        """
        Return the base UR for accessing the cluster.
        """
        return f"{self.host}{self.api_pattern}"

    @property
    def api_version(self) -> str:
        """
        Return the api version string to be used in the URL for making REST requests.
        """
        return self._api_version

    @api_version.setter
    def api_version(self, value: str) -> None:
        """
        Set the api_version string to be used in the URL for making REST requests.
        """
        if not value.startswith("v"):
            raise ValueError(
                "API version string should follow the format: 'vX', where X is the "
                "(integer) version number, e.g.: 'v1', 'v2', 'v3'."
            )
        self._api_version = value

    @property
    def api_pattern(self) -> str:
        """
        Return the API pattern used in the URL
        """
        return f"/api/{self.api_version}/"


@attrs.define(slots=False)
class ServerCredentialConfig(ServerConfig):
    """
    Configuration for an Intel® Geti™ server that requires authentication via username
    and password.

    NOTE: This is a legacy authentication method. Recent server versions should
    authenticate via a personal access token (API key)

    :var username: Username to log in to the instance.
    :var password: Password required to log in to the instance
    """

    username: str
    password: str


@attrs.define(slots=False)
class ServerTokenConfig(ServerConfig):
    """
    Configuration for an Intel® Geti™ server that uses a personal access token
    (API key) for authentication.

    :var token: Personal access token that can be used to connect to the server.
    """

    token: str


@attrs.define(slots=False)
class SaaSTokenConfig(ServerTokenConfig):
    """
    Configuration for the Intel® Geti™ SaaS environment that uses a personal access token
    (API key) for authentication.

    :var token: Personal access token that can be used to connect to the server.
    :var organization_id: ID of the organization for which the token is valid
    """

    organization_id: str
