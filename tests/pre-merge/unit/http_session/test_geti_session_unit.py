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

from typing import Any

import pytest
from pytest_mock import MockerFixture

from geti_sdk.http_session.geti_session import (
    ONPREM_MODE,
    SAAS_MODE,
    GetiSession,
    ServerTokenConfig,
)


class TestPlatformAuthentication:
    def test_authentication_saas(
        self,
        fxt_mocked_server_credential_config,
        fxt_server_token_config_parameters: dict[str, Any],
        mocker: MockerFixture,
    ):
        # Arrange
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession.platform_serving_mode",
            SAAS_MODE,
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_product_info_and_set_api_version",
            return_value={
                "product-version": "2.0.0",
                "build-version": "2.0.0-test-20240417130126",
                "smtp-defined": "True",
                "environment": "saas",
            },
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_organization_id",
            return_value="dummy_org_id",
        )
        token_server_config = ServerTokenConfig(**fxt_server_token_config_parameters)

        # Act
        # Username and password auth fails with SAAS
        with pytest.raises(ValueError):
            GetiSession(fxt_mocked_server_credential_config)

        # Token-based auth works with SAAS
        GetiSession(token_server_config)

    def test_authentication_onprem(
        self,
        fxt_mocked_server_credential_config,
        fxt_server_token_config_parameters: dict[str, Any],
        mocker: MockerFixture,
    ):
        # Arrange
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession.platform_serving_mode",
            ONPREM_MODE,
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_product_info_and_set_api_version",
            return_value={
                "product-version": "2.0.0",
                "build-version": "2.0.0-test-20240417130126",
                "smtp-defined": "True",
                "environment": "on-prem",
            },
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_organization_id",
            return_value="dummy_org_id",
        )
        mocker.patch("geti_sdk.http_session.geti_session.GetiSession.authenticate_with_password")

        # Act
        # Username and password will soon be deprecated for on-prem
        with pytest.warns():
            GetiSession(fxt_mocked_server_credential_config)
