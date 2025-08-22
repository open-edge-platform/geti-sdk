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
import glob
import logging
import os
import shutil
import time

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .constants import CASSETTE_EXTENSION


def are_cassettes_available(cassette_path) -> bool:
    """
    Checks that the VCR cassettes required to run the tests offline are available

    :return: True if the cassettes are available in the proper path, False otherwise
    """
    if not os.path.isdir(cassette_path):
        return False
    return len(os.listdir(cassette_path)) > 0


def replace_unique_entries_in_cassettes(entry_pairs: tuple[tuple[str, str]], cassette_dir: str) -> None:
    """
    This function searches for the unique_entry in a target cassette file and
    replaces all occurrences of that address by a dummy value. The cassette file is
    updated in place

    :param entry_pairs: (unique_entry, dummy_value) pairs to search for and replace
    :param cassette_dir: Path to the directory containing the cassette files to search in
    """
    logging.info(
        f"Removing tre following entries from test cassettes: '{(entry_pair[0] for entry_pair in entry_pairs)}' ..."
    )
    tqdm_prefix = "Scrubbing cassettes"

    t_start = time.time()
    cassette_paths = glob.glob(os.path.join(cassette_dir, f"*.{CASSETTE_EXTENSION}"))

    with logging_redirect_tqdm(tqdm_class=tqdm):
        for cassette_path in tqdm(cassette_paths, desc=tqdm_prefix):
            replace_unique_entries_in_cassette(entry_pairs=entry_pairs, path_to_cassette_file=cassette_path)
    logging.info(
        f"Entries scrubbed from {len(cassette_paths)} cassette files in {1000 * (time.time() - t_start)} seconds"
    )


def replace_unique_entries_in_cassette(entry_pairs: tuple[tuple[str, str]], path_to_cassette_file: str) -> None:
    """
    This function searches for the unique_entry in a target cassette file and
    replaces all occurrences of that address by a dummy value. The cassette file is
    updated in place

    :param entry_pairs: (unique_entry, dummy_value) pairs to search for and replace
    :param path_to_cassette_file: Path to the cassette file to search in
    """
    path_to_scrubbed_cassette_file = path_to_cassette_file + "_new"

    with (
        open(path_to_cassette_file) as read_file,
        open(path_to_scrubbed_cassette_file, "w") as write_file,
    ):
        for line in read_file.readlines():
            for unique_entry, dummy_value in entry_pairs:
                line = line.replace(unique_entry, dummy_value)
            write_file.write(line)

    os.remove(path_to_cassette_file)
    shutil.move(src=path_to_scrubbed_cassette_file, dst=path_to_cassette_file)
