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

from pytest_mock import MockerFixture
from requests import Response

from geti_sdk.rest_clients.credit_system_client import CreditSystemClient


class TestCreditSystemClient:
    def test_init(
        self,
        mocker: MockerFixture,
        fxt_mocked_session_factory,
    ):
        # 1. Arrange
        mock_session = fxt_mocked_session_factory()
        mock_job_id = "541"
        mock_workspace_id = "1"
        mock_job = mocker.MagicMock()
        mock_job.cost = {"cost": 10}
        mocked_get_job_by_id = mocker.patch(
            "geti_sdk.rest_clients.credit_system_client.get_job_by_id",
            return_value=mock_job,
        )

        # 2. Act and Assert
        # The CS client fires a balance call to check if the CS enabled on the server
        # If the CS enabled, the server returns a message that is decoded to a dict
        balance_response = {"available": 10, "incoming": 20, "blocked": 5}
        mocker.patch.object(mock_session, "get_rest_response", return_value=balance_response)
        cs_client = CreditSystemClient(session=mock_session, workspace_id=mock_workspace_id)
        assert cs_client.get_balance().available == balance_response["available"]

        # If the CS is disabled, the server returns an empty 200 response
        empty_response = Response()
        empty_response.status_code = 200
        mocker.patch.object(mock_session, "get_rest_response", return_value=Response())
        cs_client = CreditSystemClient(session=mock_session, workspace_id=mock_workspace_id)
        assert cs_client.get_balance() is None
        assert cs_client.get_subscriptions() is None

        # Job cost method works in any case
        cs_client.get_job_cost(mock_job_id)
        mocked_get_job_by_id.assert_called_once_with(mock_job_id, mock_session, mock_workspace_id)
