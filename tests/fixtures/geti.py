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

import pytest

from geti_sdk import Geti
from geti_sdk.http_session import GetiSession, ServerCredentialConfig, ServerTokenConfig
from tests.helpers.constants import CASSETTE_EXTENSION, DUMMY_PASSWORD, DUMMY_USER


@pytest.fixture(scope="module")
def fxt_geti_session(fxt_vcr, fxt_server_config: ServerTokenConfig | ServerCredentialConfig) -> GetiSession:
    """
    This fixture returns a GetiSession instance which has already performed
    authentication
    """
    with fxt_vcr.use_cassette(
        f"session.{CASSETTE_EXTENSION}",
        filter_post_data_parameters=[
            ("login", DUMMY_USER),
            ("password", DUMMY_PASSWORD),
        ],
        allow_playback_repeats=True,
    ):
        yield GetiSession(server_config=fxt_server_config)


@pytest.fixture(scope="module")
def fxt_geti(fxt_vcr, fxt_server_config: ServerTokenConfig | ServerCredentialConfig) -> Geti:
    """
    This fixture returns a Geti instance which has already performed
    authentication and retrieved a default workspace id
    """
    with fxt_vcr.use_cassette(
        f"geti.{CASSETTE_EXTENSION}",
        filter_post_data_parameters=[
            ("login", DUMMY_USER),
            ("password", DUMMY_PASSWORD),
        ],
    ):
        if isinstance(fxt_server_config, ServerCredentialConfig):
            auth_params = {
                "username": fxt_server_config.username,
                "password": fxt_server_config.password,
            }
        else:
            auth_params = {"token": fxt_server_config.token}
        yield Geti(
            host=fxt_server_config.host,
            verify_certificate=fxt_server_config.has_valid_certificate,
            proxies=fxt_server_config.proxies,
            **auth_params,
        )


@pytest.fixture(scope="module")
def fxt_geti_no_vcr(
    fxt_server_config: ServerTokenConfig | ServerCredentialConfig,
) -> Geti:
    if isinstance(fxt_server_config, ServerCredentialConfig):
        auth_params = {
            "username": fxt_server_config.username,
            "password": fxt_server_config.password,
        }
    else:
        auth_params = {"token": fxt_server_config.token}
    yield Geti(
        host=fxt_server_config.host,
        proxies=fxt_server_config.proxies,
        **auth_params,
        verify_certificate=fxt_server_config.has_valid_certificate,
    )
