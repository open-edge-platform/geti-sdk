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

import os
import shutil
import tempfile

import pytest


@pytest.fixture(scope="class")
def fxt_temp_directory() -> str:
    """
    This fixture returns the path to a temporary directory that can be used for
    storing data for the duration of the tests. The directory is removed after all
    tests in the test class have finished.

    :return: Path to the temporary storage directory
    """
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)


@pytest.fixture(scope="session")
def fxt_artifact_directory() -> str:
    """
    Returns the path to the directory in which test artifacts should be saved, when the
    tests are run in the CI pipeline.

    :return: Path to the directory for test artifacts
    """
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
    if os.environ.get("GITHUB_ACTIONS", False):
        yield os.environ.get("ARTIFACT_DIRECTORY", default_dir)
    else:
        yield default_dir
