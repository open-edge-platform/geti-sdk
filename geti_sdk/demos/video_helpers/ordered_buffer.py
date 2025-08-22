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
import time
from dataclasses import dataclass, field
from multiprocessing import Value
from queue import Empty, PriorityQueue
from typing import Any

import numpy as np

from geti_sdk.data_models import Prediction


@dataclass(order=True)
class IndexedResult:
    """
    Indexed container class that holds the result of model inference, with the
    original image and any additional data passed at runtime
    """

    index: int
    prediction: Prediction = field(compare=False)
    image: np.ndarray = field(compare=False)
    runtime_data: Any | None = field(compare=False, default=None)


class OrderedResultBuffer:
    """
    A buffer for inference results, ordered according to an index
    """

    def __init__(self, minsize: int = 30, maxsize: int | None = None):
        """
        Buffer inference results and assign an index, so that they can be retrieved in
        an ordered fashion.

        :param minsize: Minimal number of items to keep in the buffer. This is the
            minimal size the buffer should have before yielding items.
        :param maxsize: Maximum number of items that the buffer can contain. When the
            buffer contains this number of items, any attempt to add an item will be
            blocking.
        """
        queue_args = {"maxsize": maxsize} if isinstance(maxsize, int) else {}
        self._queue = PriorityQueue[IndexedResult](**queue_args)
        self._minsize = minsize
        self.__queue_args = queue_args
        self._put_count = Value("i", 0)
        self._get_count = Value("i", 0)

        if maxsize is not None and minsize >= maxsize:
            raise ValueError("minsize must be smaller than maxsize")

        logging.info(f"OrderedBuffer intialized with `minsize={minsize}` and `maxsize={maxsize}`")

    def put(
        self,
        index: int,
        image: np.ndarray,
        prediction: Prediction,
        runtime_data: Any | None = None,
    ):
        """
        Add an image, prediction and corresponding runtime data as an entry to the
        buffer

        :param index: The index of the result item
        :param image: The original image for which the prediction was generated
        :param prediction: The prediction for the image
        :param runtime_data: Any additional data passed at runtime
        """
        queue_item = IndexedResult(index=index, image=image, prediction=prediction, runtime_data=runtime_data)
        self._queue.put(queue_item)
        with self._put_count.get_lock():
            self._put_count.value += 1

    def get(self, timeout: float = 0, empty_buffer: bool = False) -> IndexedResult:
        """
        Get the next item from the buffer

        :param timeout: Timeout in seconds
        :param empty_buffer: True to allow retrieving all items from the buffer, i.e.
            reducing the number of available buffered items to below the minsize.
        :return: An IndexedResult object, containing the index, the original image,
            the prediction and any additional data assigned at runtime
        """
        t_start = time.time()
        item: IndexedResult | None = None
        if not empty_buffer:
            while timeout == 0 or time.time() - t_start < timeout:
                if self._queue.qsize() > self._minsize:
                    try:
                        item = self._queue.get(block=False)
                    except Empty:
                        time.sleep(1e-9)
                        continue
                    break
                time.sleep(1e-9)
        else:
            item = self._queue.get(block=True, timeout=timeout + 1e-9)
        if item is None:
            raise Empty
        self._queue.task_done()
        with self._get_count.get_lock():
            self._get_count.value += 1
        return item

    def reset(self) -> None:
        """
        Clear all remaining items from the buffer, and reset the index.
        """
        self._queue = PriorityQueue[IndexedResult](**self.__queue_args)
        with self._put_count.get_lock():
            self._put_count.value = 0
        with self._get_count.get_lock():
            self._get_count.value = 0

    @property
    def is_empty(self) -> bool:
        """
        Return True if the buffer is empty, False otherwise

        :return: True if the buffer is empty, False otherwise
        """
        return self._queue.qsize() == 0

    @property
    def total_items_buffered(self) -> int:
        """
        Return the total number of items that have been put to the buffer
        """
        with self._put_count.get_lock():
            return self._put_count.value

    @property
    def current_buffer_size(self) -> int:
        """
        Return the number of items that are currently in the buffer
        """
        with self._put_count.get_lock(), self._get_count.get_lock():
            return self._put_count.value - self._get_count.value()
