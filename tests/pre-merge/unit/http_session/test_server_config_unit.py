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

from typing import Any

import pytest

from geti_sdk.http_session import ServerCredentialConfig, ServerTokenConfig
from tests.helpers.constants import DUMMY_HOST, DUMMY_PASSWORD, DUMMY_TOKEN, DUMMY_USER


class TestServerConfig:
    def test_server_credentials_config(self, fxt_server_credential_config_parameters: dict[str, Any]):
        # Act
        server_config = ServerCredentialConfig(**fxt_server_credential_config_parameters)

        # Assert
        assert server_config.host == f"https://{DUMMY_HOST}"
        assert server_config.username == DUMMY_USER
        assert server_config.password == DUMMY_PASSWORD
        assert not server_config.has_valid_certificate
        assert server_config.proxies == fxt_server_credential_config_parameters["proxies"]

    def test_server_token_config(self, fxt_server_token_config_parameters: dict[str, Any]):
        # Act
        server_config = ServerTokenConfig(**fxt_server_token_config_parameters)

        # Assert
        assert server_config.host == f"https://{DUMMY_HOST}"
        assert server_config.token == DUMMY_TOKEN
        assert server_config.has_valid_certificate
        assert server_config.proxies is None

    def test_server_config_api_version(self, fxt_server_token_config_parameters: dict[str, Any]):
        # Arrange
        server_config = ServerTokenConfig(**fxt_server_token_config_parameters)

        # Act
        server_config.api_version = "v2"

        # Assert
        assert server_config.base_url == f"https://{DUMMY_HOST}/api/v2/"

        # Act and assert
        with pytest.raises(ValueError):
            server_config.api_version = "1"
