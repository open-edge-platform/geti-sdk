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
import datetime
import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from geti_sdk.data_models import Prediction


class PostInferenceObject(metaclass=ABCMeta):
    """
    Base interface class for post inference triggers, actions and hooks
    """

    _override_from_dict_: bool = False

    def __new__(cls, *args, **kwargs):
        """
        Create a new PostInferenceObject instance
        """
        instance = super().__new__(cls)
        instance._argument_dict_ = {"args": args, "kwargs": kwargs}
        return instance

    def __init__(self):
        self._constructor_arguments_ = self.__get_constructor_arguments()

    def __get_constructor_arguments(self) -> dict[str, Any]:
        """
        Return the arguments used for constructing the PostInferenceAction object

        :return: Dictionary containing the constructor parameter names as keys, and
            parameter values as values
        """
        constructor_argument_params = inspect.signature(self.__init__).parameters
        parameters: dict[str, Any] = {}
        args = self._argument_dict_.get("args", ())
        kwargs = self._argument_dict_.get("kwargs", {})
        for index, (pname, parameter) in enumerate(constructor_argument_params.items()):
            if index + 1 <= len(args):
                parameters.update({pname: args[index]})
            elif pname in kwargs:
                parameters.update({pname: kwargs[pname]})
            else:
                parameters.update({pname: parameter.default})
        return parameters

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the PostInferenceObject

        :return: Dictionary representing the class name and its constructor parameters
        """
        constructor_args: dict[str, Any] = {}
        for key, value in self._constructor_arguments_.items():
            if isinstance(value, PostInferenceObject):
                constructor_args.update({key: value.to_dict()})
            else:
                constructor_args.update({key: value})
        return {type(self).__name__: constructor_args}

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> "PostInferenceObject":
        """
        Construct a PostInferenceObject from an input dictionary `input_dict`

        :param input_dict: Dictionary representation of the PostInferenceObject
        :return: Instantiated PostInferenceObject, according to the input dictionary
        """
        available_objects = {subcls.__name__: subcls for subcls in cls.__subclasses__()}
        pi_objects: list[PostInferenceObject] = []
        for object_name, object_args in input_dict.items():
            target_object = available_objects[object_name]
            if target_object._override_from_dict_:
                pi_objects.append(target_object.from_dict(object_args))
            else:
                pi_objects.append(target_object(**object_args))
        return pi_objects[0]


class PostInferenceTrigger(PostInferenceObject):
    """
    Base class for post inference triggers

    Post inference triggers are used in inference hooks. They define a condition
    (or set of conditions) that determines whether any post-inference action should be
    taken.

    :param threshold: Threshold to use in activating the trigger. If the computed
        score is above the threshold, the trigger emits a positive decision and any
        downstream actions will be performed
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self._repr_info_ = f"threshold={threshold:.1f}"

    @abstractmethod
    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        return NotImplemented

    def get_decision(self, score: float) -> bool:
        """
        Make a decision based on a previously computed `score` and the threshold defined
        for the trigger

        :param score: Float representing the score for a certain image, prediction pair
        :return: Boolean indicating whether the trigger conditions are met (True,
            score is higher than the threshold), or not (False)
        """
        return score > self.threshold

    def __repr__(self) -> str:
        """
        Return string representation of the PostInferenceTrigger
        """
        return f"{type(self).__name__}({self._repr_info_})"


class PostInferenceAction(PostInferenceObject):
    """
    Base class for post inference actions. These are actions that are used in inference
    hooks, and can be (conditionally) executed after inference

    :param log_level: Log level to use in the action, current options are `info` or
        `debug`. Defaults to `debug`
    """

    def __init__(self, log_level: str = "debug"):
        super().__init__()
        if log_level.lower() == "debug":
            self.log_function = logging.debug
        elif log_level.lower() == "info":
            self.log_function = logging.info
        else:
            raise ValueError(f"Unsupported log_level `{log_level}`, options are 'info' or 'debug'.")
        self._repr_info_: str = ""

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        prediction: Prediction,
        score: float | None = None,
        name: str | None = None,
        timestamp: datetime.datetime | None = None,
    ):
        """
        Execute the action for the given `image` with corresponding `prediction` and an
        `score`. The `score` could be passed from the trigger that triggers the action.

        :param image: Numpy array representing an image
        :param prediction: Prediction object which was generated for the image
        :param score: Optional score computed from a post inference trigger
        :param name: String containing the name of the image
        :param timestamp: Datetime object containing the timestamp belonging to the
            image
        """
        return NotImplemented

    def __repr__(self) -> str:
        """
        Return string representation of the PostInferenceAction
        """
        return f"{type(self).__name__}({self._repr_info_})"


class PostInferenceHookInterface(PostInferenceObject):
    """
    Basic hook that can be executed after inference. A PostInferenceHook consists of:
      - A Trigger, defining a condition
      - An Action, defining an act or sequence of actions to perform when the
        trigger is activated.


    :param trigger: PostInferenceTrigger which is evaluated and used to trigger the
        hook action
    :param action: PostInferenceAction to be executed if triggered
    """

    def __init__(self, trigger: PostInferenceTrigger, action: PostInferenceAction):
        super().__init__()
        self.trigger = trigger
        self.action = action

    @abstractmethod
    def run(
        self,
        image: np.ndarray,
        prediction: Prediction,
        name: str | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> None:
        """
        Execute the post inference hook. This will first evaluate the `trigger`, and
        if the trigger conditions are met the corresponding `action` will be executed.

        :param image: Numpy array representing the image that was the input for
            model inference
        :param prediction: Prediction object containing the inference results for the
            image
        :param name: Optional name of the image which can be used in the hook
            action, for example as filename or tag for data collection
        :param timestamp: Optional timestamp belonging to the image
        """
        raise NotImplementedError
