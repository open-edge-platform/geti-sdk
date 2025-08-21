import time

from geti_sdk.data_models import Job, Task
from geti_sdk.data_models.algorithms import Algorithm
from geti_sdk.http_session import GetiRequestException
from geti_sdk.rest_clients import TrainingClient

from .enums import SdkTestMode


def attempt_to_train_task(
    training_client: TrainingClient,
    task: int | Task,
    test_mode: SdkTestMode = SdkTestMode.OFFLINE,
    algorithm: Algorithm | None = None,
) -> Job:
    """
    Attempts to train the `task` (either a task or the index of the task in the list
    of trainable tasks for the project), by issuing a 'train_task' command from the
    training_client.

    If the initial train request fails with a 'project_not_train_ready' error code,
    this method will attempt to issue the command again (max 5 attempts) after
    a short sleep period.

    :param training_client: TrainingClient to use to train the task
    :param task: Task or index of task to be trained
    :param test_mode: SdkTestMode in which the training is attempted. If set to
        SdkTestMode.OFFLINE, no sleep period is used between attempts
    :return: Training job that was created by the training request
    """
    job: Job | None = None
    not_ready_response = "project_not_train_ready"
    n_attempts = 10

    for i in range(n_attempts):
        try:
            job = training_client.train_task(task, algorithm=algorithm)
            break
        except GetiRequestException as error:
            if error.response_error_code != not_ready_response:
                raise error
        if test_mode != SdkTestMode.OFFLINE:
            time.sleep(5)

    if job is not None:
        return job
    raise ValueError(
        f"Unable to start training on server, received '{not_ready_response}' on "
        f"each request for {n_attempts} calls in a row. Please make sure that the "
        f"project is ready to start training the task."
    )


def await_training_start(test_mode: SdkTestMode, training_client: TrainingClient, timeout: int = 300) -> None:
    """
    Wait for the project to start training. This method will not trigger any training
    jobs, but merely await the project addressed by the `training_client` to start
    training.

    This method will poll the server in 10 second intervals, until training is started

    :param test_mode: SdkTestMode in which the training is attempted. If set to
        SdkTestMode.OFFLINE, this method does not do any waiting
    :param training_client: Training client for the project that is expected to start
        training
    :param timeout: Maximum amount of time (in seconds) to wait for training to start
    """
    if test_mode != SdkTestMode.OFFLINE:
        t_start = time.time()
        is_training = training_client.is_training()
        while not is_training and time.time() - t_start < timeout:
            is_training = training_client.is_training()
            time.sleep(10)
    return
