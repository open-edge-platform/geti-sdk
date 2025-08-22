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

from .constants import BASE_TEST_PATH


def get_sdk_fixtures() -> list[str]:
    """
    Returns the list of fixtures available to the SDK

    Recursively traverses the directory tree under the `tests/fixtures` directory.

    :return: list of fixture paths for pytest to import
    """
    fixtures_path = os.path.join(BASE_TEST_PATH, "fixtures")
    return _get_fixtures_from_folder(path_to_folder=fixtures_path, parent_folders_dot_path="tests")


def _get_fixtures_from_folder(
    path_to_folder: str | os.PathLike,
    parent_folders_dot_path: str,
    fixtures: list[str] | None = None,
) -> list[str]:
    """
    Collect fixtures from the folder at `path_to_folder`

    :param path_to_folder: Path to the folder in which to look for fixtures
    :param parent_folders_dot_path: Path (relative from the base `tests` directory` to
        the folder, in dot-notation
    :param fixtures: existing list of fixtures, in dot notation paths
    :return: list of dot notation paths corresponding to files containing fixtures
    """
    if fixtures is None:
        fixtures: list[str] = []
    for filename in os.listdir(path_to_folder):
        full_filepath = os.path.join(path_to_folder, filename)
        new_dot_path = f"{parent_folders_dot_path}.{os.path.basename(path_to_folder)}"
        if os.path.isfile(full_filepath):
            if filename.endswith(".py") and not filename.startswith("__"):
                fixtures.append(f"{new_dot_path}.{filename[0:-3]}")
        elif os.path.isdir(full_filepath):
            fixtures.extend(
                _get_fixtures_from_folder(path_to_folder=full_filepath, parent_folders_dot_path=new_dot_path)
            )
    return fixtures
