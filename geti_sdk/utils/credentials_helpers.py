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

from dotenv import dotenv_values

from geti_sdk.http_session.server_config import (
    ServerCredentialConfig,
    ServerTokenConfig,
)


def convert_boolean_env_variable(value: str | None, default_value: bool = False) -> bool:
    """
    Convert a string that was extracted from environment variables to a
    boolean.

    Strings like 'True', 'true', 't', '1', 'T' are accepted and converted to True
    Strings like 'False', 'false', 'f', '0', 'F', are accepted and converted to False
    Other strings are rejected, a ValueError will be raised in that case

    :param value: string to convert to boolean
    :param default_value: Default value to return, will be returned in case `value` is
        None
    :return: Boolean corresponding to the input `value`
    """
    true_strings = ("true", "1", "t")
    false_strings = ("false", "0", "f")

    if value is None:
        return default_value

    if value.lower() not in true_strings + false_strings:
        raise ValueError(f"Invalid value `{value}` for boolean variable")
    return value in true_strings


def get_server_details_from_env(
    env_file_path: str = ".env", use_global_variables: bool = False
) -> ServerTokenConfig | ServerCredentialConfig:
    """
    Retrieve the server information (hostname and authentication details) from
    environment variables.

    By default, this method will attempt to find a file ".env" in the current working
    directory and attempt to read the information from that file. To read from a
    different file, simply specify the path to the file in `env_file_path`. The
    following variables are relevant:

        HOST        -> hostname or ip address of the Geti server

        TOKEN       -> Personal Access Token that can be used for authorization

        VERIFY_CERT -> boolean, pass 1 or True to verify

        HTTPS_PROXY -> Address of the proxy to be used for https communication. Make
                        sure to specify the full address, including port number. For
                        example: HTTPS_PROXY=http://proxy.my-company.com:900

    In addition, authentication via credentials is also supported. In that case, the
    following variables should be provided:

        USERNAME -> username to log in to the Geti server

        PASSWORD -> password for logging in to the Geti server


    If both TOKEN, USERNAME and PASSWORD are provided, the method will use the preferred
    token authorization.

    If `use_global_variables` is set to `True`, the method attempts to retrieve the
    server details from the global environment variables instead of
    reading from file. In that case, the `GETI_` prefix should be added to the
    variable names (i.e. `HOST` becomes `GETI_HOST`, `TOKEN` becomes `GETI_TOKEN`, etc.)

    :param env_file_path: Path to the file containing the server details
    :param use_global_variables: If set to True, the method will not read the server
        details from a file but will use environment variables instead
    :return: A ServerConfig instance that contains the details of the Geti server
        specified in the environment
    """
    if use_global_variables:
        host_key = "GETI_HOST"
        token_key = "GETI_TOKEN"  # nosec: B105
        username_key = "GETI_USERNAME"
        password_key = "GETI_PASSWORD"  # nosec: B105
        cert_key = "GETI_VERIFY_CERT"
        https_proxy_key = "GETI_HTTPS_PROXY"

        retrieval_func = os.environ.get
        env_name = "environment variables"
    else:
        host_key = "HOST"
        token_key = "TOKEN"  # nosec: B105
        username_key = "USERNAME"
        password_key = "PASSWORD"  # nosec: B105
        cert_key = "VERIFY_CERT"
        https_proxy_key = "HTTPS_PROXY"

        env_variables = dotenv_values(dotenv_path=env_file_path)
        if not env_variables:
            raise ValueError(
                f"Unable to load Geti server details from environment file at path "
                f"'{env_file_path}', please make sure the file exists."
            )
        retrieval_func = env_variables.get
        env_name = "environment file"

    # Extract hostname
    hostname = retrieval_func(host_key, None)
    if hostname is None:
        raise ValueError(
            f"Unable to find Geti hostname in the {env_name}. Please make sure that the variable {host_key} is defined."
        )

    # Extract certificate validation. Defaults to True
    verify_certificate = convert_boolean_env_variable(value=retrieval_func(cert_key, None), default_value=True)

    # Extract https proxy configuration
    https_proxy = retrieval_func(https_proxy_key, None)
    proxies = {"https": https_proxy} if https_proxy is not None else None

    # Extract token/credentials
    token = retrieval_func(token_key, None)
    if token is None:
        user = retrieval_func(username_key, None)
        password = retrieval_func(password_key, None)
        if user is None or password is None:
            raise ValueError(
                f"Unable to find either personal access token or user credentials in "
                f"the {env_name}. Please make sure that either '{token_key}' or "
                f"'{username_key}' and '{password_key}' is defined in the environment."
            )
        server_config = ServerCredentialConfig(
            host=hostname,
            username=user,
            password=password,
            has_valid_certificate=verify_certificate,
            proxies=proxies,
        )
    else:
        server_config = ServerTokenConfig(
            host=hostname,
            token=token,
            has_valid_certificate=verify_certificate,
            proxies=proxies,
        )
    return server_config
