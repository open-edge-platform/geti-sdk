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

from collections.abc import Callable
from typing import Any

import pytest
from pytest_mock import MockerFixture

from geti_sdk import Geti
from geti_sdk.http_session import GetiSession, ServerCredentialConfig
from geti_sdk.http_session.geti_session import ONPREM_MODE
from geti_sdk.http_session.server_config import ServerConfig


@pytest.fixture
def fxt_mocked_server_credential_config():
    yield ServerCredentialConfig(host="dummy_host", username="dummy_user", password="dummy_password")


@pytest.fixture
def fxt_mocked_session_factory(
    mocker: MockerFixture, fxt_mocked_server_credential_config: ServerCredentialConfig
) -> Callable[[Any], GetiSession]:
    def _mocked_session_factory(
        return_value: list | dict | None = None,
        server_config: ServerConfig | None = None,
    ) -> GetiSession:
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession.platform_serving_mode",
            ONPREM_MODE,
        )
        mocker.patch("geti_sdk.http_session.geti_session.GetiSession.authenticate_with_password")
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession.get_rest_response",
            return_value=return_value,
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_product_info_and_set_api_version",
            return_value={
                "product-version": "2.3.0",
                "build-version": "2.3.0-test-20240417130126",
                "smtp-defined": "True",
                "environment": "on-prem",
            },
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_organization_id",
            return_value="000000000000000000000001",
        )
        if server_config is None:
            server_config = fxt_mocked_server_credential_config
        return GetiSession(server_config=server_config)

    yield _mocked_session_factory


@pytest.fixture
def fxt_mocked_geti(mocker: MockerFixture, fxt_mocked_session_factory):
    mocker.patch("geti_sdk.geti.get_workspace_id", return_value=1)
    mocker.patch("geti_sdk.geti.GetiSession", new=fxt_mocked_session_factory)
    geti = Geti(host="dummy_host", password="dummy_password", username="dummy_username")

    yield geti
