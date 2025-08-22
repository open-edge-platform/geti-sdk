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

import logging
from datetime import datetime

from geti_sdk.data_models import CreditAccount, CreditBalance, Subscription
from geti_sdk.data_models.job import Job, JobCost
from geti_sdk.http_session import GetiSession
from geti_sdk.utils.job_helpers import get_job_by_id
from geti_sdk.utils.serialization_helpers import deserialize_dictionary
from geti_sdk.utils.workspace_helpers import get_workspace_id


def allow_supported(func):
    """
    Decorate the class methods to allow them to run only if the Credit System is supported.
    """

    def wrapper(instance, *args, **kwargs):
        if instance._is_supported:
            return func(instance, *args, **kwargs)
        logging.warning("Credit System is not supported by the Intel Geti Platform.")
        return None

    return wrapper


class CreditSystemClient:
    """
    Class to work with credits in Intel Geti.
    """

    def __init__(self, session: GetiSession, workspace_id: str | None = None):
        self.session = session
        if workspace_id is not None:
            self.workspace_id = workspace_id
        else:
            self.workspace_id = get_workspace_id(self.session)
        # Make sure the Intel Geti Platform supports Credit System.
        self._is_supported = self.is_supported()

    def is_supported(self) -> bool:
        """
        Check if the Intel Geti Platform supports Credit system.

        :return: True if the Credit System is supported, False otherwise.
        """
        # Send a GET request to the balance endpoint to check if the Credit System is supported.
        # The text response is allowed to check if the server responds with a default
        # html page in case the Credit System is not supported.
        r = self.session.get_rest_response(
            url=self.session.base_url + "balance",
            method="GET",
            allow_text_response=True,
        )
        # If the Platform responds with the information about the available subscriptions,
        # then it supports Credit System. Session will decode the message into a `dict`
        # In case the server returns an empty 200 message, it means the Credit System is not supported.
        # The session would return a Response object.
        return isinstance(r, dict)

    @allow_supported
    def get_balance(self, timestamp: datetime | None = None) -> CreditBalance | None:
        """
        Get the current credit balance in the workspace.

        :param timestamp: The timestamp to get the balance at. If None, the current balance is returned.
        :return: The available credit balance in the workspace.
        """
        query_postfix = f"?date={int(timestamp.timestamp() * 1000)}" if timestamp else ""
        response = self.session.get_rest_response(
            url=self.session.base_url + "balance" + query_postfix,
            method="GET",
        )
        return deserialize_dictionary(response, CreditBalance)

    def get_job_cost(self, job: Job | str) -> JobCost | None:
        """
        Get the cost of a job.

        This method allows you to find out the cost of a training or an optimization job.

        :param job: A Job object or a Job ID.
        :return: A JobCost object presenting the total cost and the consumed credits.
        """
        job_id = job.id if isinstance(job, Job) else job
        fetched_job = get_job_by_id(job_id, self.session, self.workspace_id)
        return fetched_job.cost if fetched_job else None

    @allow_supported
    def get_subscriptions(self) -> list[Subscription] | None:
        """
        Get the subscription details for the workspace.

        :return: A list of Subscription objects.
        """
        response = self.session.get_rest_response(
            url=self.session.base_url + "subscriptions",
            method="GET",
        )
        return [deserialize_dictionary(sub, Subscription) for sub in response.get("subscriptions", [])]

    @allow_supported
    def get_credit_accounts(self) -> list[CreditAccount] | None:
        """
        Get the subscription details for the workspace.

        :return: A list of Subscription objects.
        """
        response = self.session.get_rest_response(
            url=self.session.base_url + "credit_accounts",
            method="GET",
        )
        return [deserialize_dictionary(sub, CreditAccount) for sub in response.get("credit_accounts", [])]
