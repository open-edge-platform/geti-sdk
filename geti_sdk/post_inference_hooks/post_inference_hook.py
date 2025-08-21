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
import copy
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any

import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import (
    PostInferenceAction,
    PostInferenceHookInterface,
    PostInferenceTrigger,
)
from geti_sdk.post_inference_hooks.utils import RateLimiter


class PostInferenceHook(PostInferenceHookInterface):
    """
    Basic hook that can be executed after inference. A PostInferenceHook consists of:
      - A Trigger, defining a condition
      - An Action, defining an act or sequence of actions to perform when the
        trigger is activated.

    By default, hooks are executed in parallel in a separate thread from the main
    inference code. If you want to disable parallel execution, make sure to set
    `max_threads=0` when creating the hook.

    :param trigger: PostInferenceTrigger which is evaluated and used to trigger the
        hook action
    :param action: PostInferenceAction to be executed if triggered
    :param max_threads: Maximum number of threads to use for hook execution. Defaults
        to 5. Set to 0 to disable parallel hook execution.
    :param limit_action_rate: True to limit the rate at which the `action` for the
        hook is executed. Note that rate limiting is applied **after** trigger
        evaluation, so this can be useful for handling video streaming in which
        subsequent frames may be very similar. Enabling rate limiting will avoid
        sampling many near-duplicate frames
    :param max_frames_per_second: Maximum frame rate at which to execute the action.
        Only takes effect if `limit_action_rate = True`.
    :param queue_limit: Maximum size of the execution queue for the hook. If left as 0
        (the default), the queue is unbounded. This may lead to high memory usage in
        some cases (if inference is very fast, or the action execution is slow). When
        running into out of memory issues, try setting the queue limit to avoid
        buffering too many frames in memory. Note that this may slow down the
        inference rate
    """

    def __init__(
        self,
        trigger: PostInferenceTrigger,
        action: PostInferenceAction,
        max_threads: int = 5,
        limit_action_rate: bool = False,
        max_frames_per_second: float = 1,
        queue_limit: int = 0,
    ):
        super().__init__(trigger=trigger, action=action)

        self.parallel_execution = max_threads != 0
        self._semaphore: BoundedSemaphore | None = None

        if self.parallel_execution:
            self.executor = ThreadPoolExecutor(max_workers=max_threads)
            logging.debug(f"Parallel inference hook execution enabled, using a maximum of {max_threads} threads.")
            if queue_limit > 0:
                self._semaphore = BoundedSemaphore(queue_limit + max_threads)

        if limit_action_rate:
            self.rate_limiter: RateLimiter | None = RateLimiter(
                frames_per_second=max_frames_per_second, is_blocking=False
            )
        else:
            self.rate_limiter = None

    def run(
        self,
        image: np.ndarray,
        prediction: Prediction,
        name: str | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> None:
        """
        Run the post inference hook. First evaluate the trigger, then execute the
        action if the trigger conditions are met.

        :param image: Numpy array representing the image
        :param prediction: Prediction object containing the inference results for the
            image
        :param name: Optional name of the image which can be used in the hook
            action, for example as filename or tag for data collection
        :param timestamp: Optional timestamp belonging to the image
        """

        def execution_function(im: np.ndarray, pred: Prediction, ts: datetime.datetime) -> None:
            score = self.trigger(image=im, prediction=pred)
            decision = self.trigger.get_decision(score=score)
            if decision:
                take_action = next(self.rate_limiter) if self.rate_limiter is not None else True
                if take_action:
                    self.action(image=im, prediction=pred, score=score, name=name, timestamp=ts)

        if timestamp is None:
            timestamp = datetime.datetime.now()
        if self.parallel_execution:
            if self._semaphore is not None:
                self._semaphore.acquire()
            try:
                future = self.executor.submit(execution_function, image, prediction, timestamp)
            except Exception as error:
                if self._semaphore is not None:
                    self._semaphore.release()
                raise error
            if self._semaphore is not None:
                future.add_done_callback(lambda x: self._semaphore.release())
        else:
            execution_function(image, prediction, timestamp)

    def __del__(self):
        """
        Await shutdown of the ThreadPoolExecutor, if needed.

        Called upon deletion of the post inference hook or garbage collection
        """
        if self.parallel_execution:
            self.executor.shutdown(wait=True)

    def __repr__(self) -> str:
        """
        Return a string representation of the PostInferenceHook object

        :return: String representing the post inference hook
        """
        rate_msg = ""
        if self.rate_limiter is not None:
            rate_msg = f"Action rate limited to {1 / self.rate_limiter.interval:.1f} frames per second."
        thread_msg = ""
        if self.parallel_execution:
            thread_msg = "Multithreaded execution enabled."

        suffix_msg = f"[{thread_msg} {rate_msg}]"
        if len(suffix_msg) == 3:
            suffix_msg = ""
        return f"PostInferenceHook(trigger={self.trigger}, action={self.action}){suffix_msg}"

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> "PostInferenceHook":
        """
        Create a PostInferenceHook object from it's dictionary representation

        :param input_dict: Dictionary representing the hook
        :return: Initialized PostInferenceHook object, with a trigger and action set
            according to the input dictionary
        """
        input_copy = copy.deepcopy(input_dict)

        # Construct trigger
        trigger = PostInferenceTrigger.from_dict(input_copy["trigger"])
        input_copy.update({"trigger": trigger})

        # Construct action
        action = PostInferenceAction.from_dict(input_copy["action"])
        input_copy.update({"action": action})

        return cls(**input_copy)
