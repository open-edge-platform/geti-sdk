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

import pytest
from vcr import VCR

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.rest_clients import ProjectClient
from tests.helpers import ProjectService, force_delete_project
from tests.helpers.constants import CASSETTE_EXTENSION
from tests.helpers.enums import SdkTestMode


@pytest.fixture(scope="class")
def fxt_project_client(fxt_geti: Geti) -> ProjectClient:
    """
    This fixture returns a ProjectClient instance corresponding to the Geti instance
    """
    yield ProjectClient(session=fxt_geti.session, workspace_id=fxt_geti.workspace_id)


@pytest.fixture(scope="class")
def fxt_project_client_no_vcr(fxt_geti_no_vcr: Geti) -> ProjectClient:
    """
    This fixture returns a ProjectClient instance corresponding to the Geti instance
    """
    yield ProjectClient(session=fxt_geti_no_vcr.session, workspace_id=fxt_geti_no_vcr.workspace_id)


@pytest.fixture(scope="class")
def fxt_project_service(
    fxt_vcr,
    fxt_test_mode,
    fxt_geti: Geti,
) -> ProjectService:
    """
    This fixture provides a service for creating a project and the corresponding
    clients to interact with it.

    A project can be created by using `fxt_project_service.create_project()`, which
    takes various parameters

    The project is deleted once the test function finishes.
    """
    project_service = ProjectService(geti=fxt_geti, vcr=fxt_vcr, is_offline=(fxt_test_mode == SdkTestMode.OFFLINE))
    yield project_service
    project_service.delete_project()


@pytest.fixture(scope="class")
def fxt_project_service_2(
    fxt_vcr,
    fxt_test_mode,
    fxt_geti: Geti,
) -> ProjectService:
    """
    This fixture provides a service for creating a project and the corresponding
    clients to interact with it.

    A project can be created by using `fxt_project_service.create_project()`, which
    takes various parameters

    The project is deleted once the test function finishes.

    NOTE: This fixture is the same as `fxt_project_service`, but was added to make
    it possible to persist two projects for the scope of one test class
    """
    project_service = ProjectService(geti=fxt_geti, vcr=fxt_vcr, is_offline=(fxt_test_mode == SdkTestMode.OFFLINE))
    yield project_service
    project_service.delete_project()


@pytest.fixture(scope="class")
def fxt_project_service_no_vcr(fxt_geti_no_vcr: Geti) -> ProjectService:
    project_service = ProjectService(geti=fxt_geti_no_vcr, vcr=None)
    yield project_service
    # project_service.delete_project()


@pytest.fixture(scope="class")
def fxt_project_finalizer(fxt_project_client: ProjectClient) -> Callable[[str], None]:
    """
    This fixture returns a finalizer to ensure project deletion

    :var project_name: Name of the project for which to add the finalizer
    """

    def _project_finalizer(project: Project) -> None:
        force_delete_project(project, fxt_project_client)

    return _project_finalizer


@pytest.fixture(scope="function")
def fxt_existing_projects(fxt_vcr: VCR, fxt_project_client: ProjectClient) -> list[Project]:
    """
    This fixture returns a list of the projects that currently exist on the Geti server
    """
    with fxt_vcr.use_cassette(f"existing_projects.{CASSETTE_EXTENSION}"):
        return fxt_project_client.get_all_projects()
