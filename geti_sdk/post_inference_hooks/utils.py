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
"""Utility functions for the post inference hook module"""

import time
from threading import Lock

try:
    from collections.abc import Iterator
except ImportError:
    # In python < 3.10 Iterator is in 'collections'
    from collections.abc import Iterator


class RateLimiter(Iterator):
    """
    Iterator that yields at a maximum frequency of `frames_per_second`.

    :param frames_per_second: Maximum execution rate, defined in frames per second
    :param is_blocking: Setting this to True will block execution until the next
        yield. Setting this to False will cause the Iterator to yield `False` if it
        is called before the next yield is available, and True otherwise
    """

    def __init__(self, frames_per_second: float = 1, is_blocking: bool = False) -> None:
        self.interval = 1 / frames_per_second
        self.is_blocking = is_blocking
        self.__next_yield = 0
        self.__lock = Lock()

    def __iter__(self):
        """
        Return the iterator
        """
        return self

    def __next__(self) -> bool:
        """
        Return `True` if called after the `self.interval` has elapsed with respect to
        the last call. If `self.is_blocking == True`, will block execution otherwise,
        until the interval has elapsed. If `self.is_blocking == False`, execution is
        not blocked but `False` is returned instead.
        """
        with self.__lock:
            t = time.monotonic()
            if t < self.__next_yield:
                if self.is_blocking:
                    time.sleep(self.__next_yield - t)
                    t = time.monotonic()
                else:
                    return False
            self.__next_yield = t + self.interval
            return True
