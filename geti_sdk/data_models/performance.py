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


import attr


@attr.define()
class Score:
    """
    Score attribute in TaskPerformance.

    :var value: Value of the score
    :var metric_type: The type of matrix the score represents
    """

    value: float | None = None
    metric_type: str | None = None


@attr.define()
class TaskPerformance:
    """
    Task Performance metrics in Intel® Geti™.

    :var task_id: Unique ID of the task to which this Performance metric
        applies.
    :var score: Score of the project or model for each task
    :var local_score: Accuracy of the model or project with respect to object
        localization for each task
    :var global_score: Accuracy of the model or project with respect to global
        classification of the full image for each task
    """

    task_id: str | None = None
    score: Score | None = None
    local_score: Score | None = None
    global_score: Score | None = None


@attr.define()
class Performance:
    """
    Performance metrics for a project or model in Intel® Geti™.

    :var score: Overall score of the project or model
    :var local_score: Accuracy of the model or project with respect to object
        localization
    :var global_score: Accuracy of the model or project with respect to global
        classification of the full image
    """

    score: float | None = None
    local_score: float | None = None
    global_score: float | None = None
    task_performances: list[TaskPerformance] | None = None
