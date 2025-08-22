# Copyright (C) 2023 Intel Corporation
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
import os
import queue
import threading

import numpy as np

from geti_sdk.http_session import GetiRequestException
from geti_sdk.rest_clients import ImageClient


class Uploader:
    """
    Upload images to Intel Geti Platform using threading.

    :param num_worker_threads: Number of workers to use for the uploading
    :param image_client: ImageClient instance that will be used to upload the images
    """

    def __init__(self, num_worker_threads: int, image_client: ImageClient) -> None:
        self.num_worker_threads = num_worker_threads
        self.image_client = image_client
        self.q = queue.Queue()
        self.threads = []
        self.run = True

        for i in range(num_worker_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)

    def upload_image(self, image: np.ndarray | str | os.PathLike):
        """
        Upload image to the Intel Geti project

        :param image: Image to upload
        """
        try:
            self.image_client.upload_image(image)
        except GetiRequestException as error:
            logging.exception(f"Error: Upload failed with error '{error}'")

    def worker(self):
        """
        Get item from the queue and upload it
        """
        process = True
        while process:
            if self.run:
                item = self.q.get()
                if item is None:
                    process = False
                else:
                    self.upload_image(item)
                    self.q.task_done()

    def add_data(self, data: np.ndarray | str | os.PathLike):
        """
        Add image data to upload to the queue

        :param data: Image array or image file to upload
        """
        self.q.put(data)

    @property
    def queue_length(self) -> int:
        """
        Return the current queue length

        :return: Length of the queue
        """
        return self.q.qsize()

    def stop(self):
        """
        Stop workers
        """
        for i in range(self.num_worker_threads):
            self.q.put(None)
        for t in self.threads:
            t.join()
