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

import hashlib
import logging
import os
import shutil
import urllib.request
import zipfile

import requests
from tqdm import tqdm


def get_proxies(url: str = "", verify_cert: bool = True) -> dict[str, str]:
    """
    Determine whether or not to use proxies to attempt to reach a certain url.

    :param url: URL that should be resolved
    :param verify_cert: False to disable SSL certificate validation
    :return:
    """
    logging.info(f"Connecting to url {url}...")
    proxies: dict[str, str] = {}
    timeout = 10
    try:
        requests.head(url=url, proxies=proxies, timeout=timeout, verify=verify_cert)
        return proxies
    except requests.exceptions.ConnectionError:
        logging.info("Unable to reach URL, attempting to connect via proxy...")
    proxies = urllib.request.getproxies()
    logging.debug(f"Proxies configuration: {proxies}")
    try:
        requests.head(url=url, proxies=proxies, verify=verify_cert, timeout=timeout)
        logging.info("Connection succeeded.")
    except requests.exceptions.ConnectionError as error:
        raise ValueError(
            "Unable to resolve URL with any proxy settings, please try to turn off "
            "your VPN connection before attempting again."
        ) from error
    return proxies


def download_file(
    url: str,
    target_folder: str | None,
    check_valid_archive: bool = False,
    verify_cert: bool = True,
    timeout: int = 1800,
) -> str:
    """
    Download a file from `url` to a folder on local disk `target_folder`.

    NOTE: If a file with the same name as the file to be downloaded already exists in
        `target_folder`, this function will not download anything. If
        `check_valid_archive` is True, this function not only checks if the target
        file exists but also if it is a valid .zip archive.

    :param url:
    :param target_folder:
    :param check_valid_archive: Check if the target file is a valid zip archive
    :param verify_cert: False to disable SSL certificate validation
    :param timeout: Time (in seconds) after which the download will time out
    :return: path to the downloaded file
    """
    filename = url.split("/")[-1]
    if target_folder is None:
        target_folder = "data"
    path_to_file = os.path.join(target_folder, filename)
    valid_file_exists = False
    if os.path.exists(path_to_file) and os.path.isfile(path_to_file):
        valid_file_exists = True
        if check_valid_archive and not zipfile.is_zipfile(path_to_file):
            logging.info(
                f"File {filename} exists at {path_to_file}, but is is not a valid "
                f"archive. Overwriting the existing file."
            )
            try:
                shutil.rmtree(path_to_file)
            except NotADirectoryError:
                os.remove(path_to_file)
            valid_file_exists = False
    if valid_file_exists:
        logging.info(f"File {filename} exists at {path_to_file}. No new data was downloaded.")
        return path_to_file

    proxies = get_proxies(url, verify_cert=verify_cert)
    logging.info(f"Downloading {filename}...")
    with requests.get(url, stream=True, proxies=proxies, verify=verify_cert, timeout=timeout) as r:
        if r.status_code != 200:
            r.raise_for_status()
            raise RuntimeError(f"Request to {url} failed, returned status code {r.status_code}")
        file_size = int(r.headers.get("Content-Length", 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc="") as r_raw, open(path_to_file, "wb") as f:
            shutil.copyfileobj(r_raw, f)
    logging.info("Download complete.")
    return path_to_file


def validate_hash(file_path: str, expected_hash: str) -> None:
    """
    Verify that hash matches the calculated hash of the file.

    :param file_path: Path to file.
    :param expected_hash: Expected hash of the file.
    """
    with open(file_path, "rb") as hash_file:
        downloaded_hash = hashlib.sha3_512(hash_file.read()).hexdigest()
    if downloaded_hash != expected_hash:
        raise ValueError(f"Downloaded file {file_path} does not match the required hash.")


def set_directory_permissions(target_directory: str, file_permissions=0o660, dir_permissions=0o770) -> None:
    """
    Set read, write, execute permissions for user and user_group on a directory tree.

    NOTE: Please use this method with caution, on temporary directories only

    :param target_directory: path to the directory for which to set the permissions
    :param file_permissions: Permissions to apply to all files found in the directory
        tree
    :param dir_permissions: Permissions to apply to all directories found in the
        directory tree
    """
    os.chmod(target_directory, dir_permissions)  # nosec: B103
    for root, dirs, files in os.walk(target_directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.chmod(file_path, file_permissions)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.chmod(dir_path, dir_permissions)  # nosec: B103
