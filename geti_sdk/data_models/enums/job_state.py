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

from enum import Enum


class JobState(Enum):
    """
    Enum representing the state of a job on the Intel® Geti™ server.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INACTIVE = "inactive"
    SCHEDULED = "scheduled"

    def __str__(self) -> str:
        """
        Return the string representation of the JobState instance.
        """
        return self.value

    @classmethod
    def active_states(cls) -> list["JobState"]:
        """
        Return a list of JobState instance which represent jobs that are still active.

        :return: List of JobState instances
        """
        return [cls.IDLE, cls.PAUSED, cls.RUNNING]

    @classmethod
    def inactive_states(cls) -> list["JobState"]:
        """
        Return a list of JobState instance which represent jobs that are inactive,
        i.e. cancelled, errored or finished successfully.

        :return: List of JobState instances
        """
        return [cls.FINISHED, cls.ERROR, cls.FAILED, cls.FINISHED, cls.INACTIVE]
