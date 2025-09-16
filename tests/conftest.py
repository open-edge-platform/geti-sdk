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
"""This module defines the test configuration"""

import logging
import os
import shutil
import tempfile

import pytest
from _pytest.main import Session

from geti_sdk import Geti
from geti_sdk.http_session import ServerCredentialConfig, ServerTokenConfig
from geti_sdk.utils.workspace_helpers import MultipleWorkspacesException
from tests.helpers.project_helpers import remove_all_test_projects

from .helpers import SdkTestMode, get_sdk_fixtures, replace_unique_entries_in_cassettes
from .helpers.constants import (
    BASE_TEST_PATH,
    CASSETTE_BASE_PATH,
    CASSETTE_PATH_KEY,
    DUMMY_HOST,
    DUMMY_ORGANIZATION_ID,
    DUMMY_WORKSPACE_ID,
    RECORD_CASSETTE_KEY,
)

pytest_plugins = get_sdk_fixtures()

# -------------------------------------------------------
# ---------------- Environment variables ----------------
# -------------------------------------------------------

TEST_MODE = SdkTestMode[os.environ.get("TEST_MODE", "OFFLINE")]
# TEST_MODE specifies the mode in which tests are run. Only applies to the integration
# tests. Possible modes are: "OFFLINE", "ONLINE", "RECORD"

GETI_PLATFORM_VERSION = os.environ.get("GETI_PLATFORM_VERSION", "DEVELOP")
# GETI_PLATFORM_VERSION specifies the version of the Geti platform to run the tests
# against. This is only used in the nightly tests in "OFFLINE" and "RECORD" modes.
# Possible values: "DEVELOP", "LEGACY", "SAAS".

HOST = os.environ.get("GETI_HOST", "https://dummy_host").strip("/")
# HOST should hold the domain name or ip address of the Geti instance to run the tests
# against.

USERNAME = os.environ.get("GETI_USERNAME", "dummy_user")
# USERNAME should hold the username that is used for logging in to the Geti instance

PASSWORD = os.environ.get("GETI_PASSWORD", "dummy_password")
# PASSWORD should hold the password that is used for logging in to the Geti instance

TOKEN = os.environ.get("GETI_TOKEN", None)
# TOKEN should hold the Personal Access Token that can be used to access the server.
# When both a TOKEN and username + password are provided, the test suite will use the
# TOKEN to execute the tests

GETI_HTTP_PROXY = os.environ.get("GETI_HTTP_PROXY_URL", None)
GETI_HTTPS_PROXY = os.environ.get("GETI_HTTPS_PROXY_URL", None)
# GETI_HTTP_PROXY and GETI_HTTPS_PROXY are urls to the proxy servers that should be
# used to connect to the Geti instance.
# NOTE: PROXIES can only be used in ONLINE mode, they cannot be used in RECORD mode
# (will raise an error) and have no effect in OFFLINE mode.

CLEAR_EXISTING_TEST_PROJECTS = os.environ.get("CLEAR_EXISTING_TEST_PROJECTS", "0").lower() in ["true", "1"]
# CLEAR_EXISTING_TEST_PROJECTS is a boolean that determines whether existing test
# projects are deleted before a test run

NIGHTLY_TEST_LEARNING_PARAMETER_SETTINGS = os.environ.get("LEARNING_PARAMETER_SETTINGS", "default")
# NIGHTLY_TEST_LEARNING_PARAMETER_SETTINGS determines how the learning parameters are
# set for the nightly tests. Possible values are:
#   - "default"     : The default settings
#   - "minimal"     : Single epoch and batch_size of 1
#   - "reduced_mem" : Default epochs, but batch_size of 1

# ------------------------------------------
# ---------------- Fixtures ----------------
# ------------------------------------------


@pytest.fixture(scope="session")
def fxt_server_config() -> ServerTokenConfig | ServerCredentialConfig:
    """
    This fixture holds the login configuration to access the Geti server
    """
    # Configure proxies for OFFLINE, ONLINE and RECORD mode
    if TEST_MODE == SdkTestMode.OFFLINE:
        proxies = {"https": "", "http": ""}
    elif TEST_MODE == SdkTestMode.ONLINE and (GETI_HTTP_PROXY is not None or GETI_HTTPS_PROXY is not None):
        proxies = {"https": GETI_HTTPS_PROXY, "http": GETI_HTTP_PROXY}
    else:
        # In RECORD mode or when both proxies are None, the `proxies` argument for the
        # ServerConfig should be set to None
        proxies = None
        if TEST_MODE == SdkTestMode.RECORD and (GETI_HTTPS_PROXY is not None or GETI_HTTP_PROXY is not None):
            raise ValueError(
                "Unable to use proxy servers in RECORD mode! Please clear the "
                "GETI_HTTPS_PROXY and GETI_HTTP_PROXY environment variables before "
                "running the test suite in RECORD mode."
            )

    # Use token if available
    if TOKEN is None:
        test_config = ServerCredentialConfig(host=HOST, username=USERNAME, password=PASSWORD, proxies=proxies)
    else:
        test_config = ServerTokenConfig(host=HOST, token=TOKEN, proxies=proxies)
    yield test_config


@pytest.fixture(scope="session")
def fxt_base_test_path() -> str:
    """
    This fixture returns the absolute path to the `tests` folder
    """
    yield BASE_TEST_PATH


@pytest.fixture(scope="session")
def fxt_test_mode() -> SdkTestMode:
    """
    This fixture returns the SdkTestMode with which the tests are run
    :return:
    """
    yield TEST_MODE


@pytest.fixture(scope="session")
def fxt_learning_parameter_settings() -> str:
    """
    This fixture returns the settings to use in the nightly test learning parameters.
    Can be either:
      - 'minimal' (single epoch, batch_size = 1),
      - 'reduced_mem' (normal epochs but reduced batch size for memory hungry algos)
      - 'default' (default settings)

    :return:
    """
    yield NIGHTLY_TEST_LEARNING_PARAMETER_SETTINGS


@pytest.fixture(scope="session")
def fxt_github_actions_environment() -> bool:
    """
    Return True if the tests are running in a GitHub actions environment, False
    otherwise
    """
    yield os.environ.get("GITHUB_ACTIONS", False)


# ----------------------------------------------
# ---------------- Pytest hooks ----------------
# ----------------------------------------------


def _get_geti_instance() -> Geti:
    """
    This function returns a Geti instance with the correct authentication parameters
    """
    # Handle authentication via token or credentials
    auth_params = {"username": USERNAME, "password": PASSWORD} if TOKEN is None else {"token": TOKEN}
    # Handle proxies
    if GETI_HTTP_PROXY is not None or GETI_HTTPS_PROXY is not None:
        proxies = {"http": GETI_HTTP_PROXY, "https": GETI_HTTPS_PROXY}
    else:
        proxies = None

    # handle multiple workspaces
    try:
        return Geti(host=HOST, **auth_params, proxies=proxies, verify_certificate=False)
    except MultipleWorkspacesException as exe:
        workspaces_list = exe.ws_ids_and_names
        workspace_id = workspaces_list[0][0]
        print(f"workspace id {workspace_id}")
        return Geti(host=HOST, **auth_params, proxies=proxies, verify_certificate=False, workspace_id=workspace_id)


def pytest_sessionstart(session: Session) -> None:
    """
    This function is called before a pytest test run begins.

    If the tests are run in record mode, this hook sets up a temporary directory to
    record the new cassettes to.

    :param session: Pytest session instance that has just been created
    """
    versioned_cassette_path = os.path.join(CASSETTE_BASE_PATH, GETI_PLATFORM_VERSION)
    if not os.path.exists(versioned_cassette_path):
        os.makedirs(versioned_cassette_path)
    os.environ.update({CASSETTE_PATH_KEY: versioned_cassette_path})
    if CLEAR_EXISTING_TEST_PROJECTS and TEST_MODE != SdkTestMode.OFFLINE:
        # Remove existing test projects
        remove_all_test_projects(geti=_get_geti_instance())
    if TEST_MODE == SdkTestMode.RECORD:
        record_cassette_path = tempfile.mkdtemp()
        logging.info(f"Cassettes will be recorded to `{record_cassette_path}`.")
        os.environ.update({RECORD_CASSETTE_KEY: record_cassette_path})


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    """
    This function is called after a pytest test run finishes.

    If the tests are run in record mode, this hook handles saving and cleaning the
    recorded cassettes

    :param session: Pytest session that has just finished
    :param exitstatus: Exitstatus with which the tests completed
    """
    if TEST_MODE == SdkTestMode.RECORD:
        record_cassette_path = os.environ.pop(RECORD_CASSETTE_KEY)
        versioned_cassette_path = os.environ.pop(CASSETTE_PATH_KEY)
        if exitstatus == 0:
            logging.info("Recording successful! Wrapping up....")
            # Scrub hostname, organization_id and workspace_id from cassettes
            geti = _get_geti_instance()
            organization_id_pair = (
                geti.session._organization_id,
                DUMMY_ORGANIZATION_ID,
            )
            workspace_id_pair = (geti.workspace_id, DUMMY_WORKSPACE_ID)
            host_name_pair = (HOST.replace("https://", "").strip("/"), DUMMY_HOST)

            # replace_host_name_in_cassettes(HOST, cassette_dir=record_cassette_path)
            replace_unique_entries_in_cassettes(
                (organization_id_pair, workspace_id_pair, host_name_pair),
                cassette_dir=record_cassette_path,
            )
            logging.info(
                " Hostname, organization and workspace ids were scrubbed from all cassette files successfully."
            )
            # Copy recorded cassettes to fixtures/cassettes
            logging.info(
                f"Copying newly recorded cassettes from `{record_cassette_path}` to `{versioned_cassette_path}`."
            )
            for root, dirs, files in os.walk(record_cassette_path):
                for file in files:
                    shutil.move(
                        os.path.join(root, file),
                        os.path.join(versioned_cassette_path, file),
                    )

        else:
            # Clean up any cassettes already recorded
            if os.path.exists(record_cassette_path):
                shutil.rmtree(record_cassette_path)
