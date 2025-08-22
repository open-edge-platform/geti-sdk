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

import time

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.data_models.enums import JobState
from geti_sdk.rest_clients import PredictionClient, TrainingClient


def ensure_project_is_trained(geti: Geti, project: Project) -> bool:
    """
    Ensure that the `project` has a trained model for each task.

    If no trained model is found for any of the tasks, the function will attempt to
    start training for that task. It will then await the completion of the training job.

    This method returns True if all tasks in the project have a trained model
    available, and the project is therefore ready to make predictions.

    :param geti: Geti instance pointing to the GETi server
    :param project: Project object, representing the project in GETi
    :return: True if the project is trained and ready to make predictions, False
        otherwise
    """
    prediction_client = PredictionClient(session=geti.session, workspace_id=geti.workspace_id, project=project)
    if prediction_client.ready_to_predict:
        print(f"\nProject '{project.name}' is ready to predict.\n")
        return True

    print(f"\nProject '{project.name}' is not ready for prediction yet, awaiting model training completion.\n")
    training_client = TrainingClient(session=geti.session, workspace_id=geti.workspace_id, project=project)
    # If there are no jobs running for the project, we launch them
    jobs = training_client.get_jobs(project_only=True)
    running_jobs = [job for job in jobs if job.state in [JobState.RUNNING, JobState.SCHEDULED]]
    tasks = project.get_trainable_tasks()

    new_jobs = []
    if len(running_jobs) != len(tasks):
        for task in project.get_trainable_tasks():
            new_jobs.append(training_client.train_task(task))

    # Monitor job progress to ensure training finishes
    training_client.monitor_jobs(running_jobs + new_jobs)

    tries = 20
    while not prediction_client.ready_to_predict and tries > 0:
        time.sleep(1)
        tries -= 1

    if prediction_client.ready_to_predict:
        print(f"\nProject '{project.name}' is ready to predict.\n")
        prediction_ready = True
    else:
        print(
            f"\nAll jobs completed, yet project '{project.name}' is still not ready "
            f"to predict. This is likely due to an error in the training job.\n"
        )
        prediction_ready = False
    return prediction_ready
