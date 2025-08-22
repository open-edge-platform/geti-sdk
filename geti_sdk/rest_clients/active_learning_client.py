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
from geti_sdk.data_models import Image, Project, Video
from geti_sdk.data_models.containers import MediaList
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters import MediaRESTConverter


class ActiveLearningClient:
    """
    Class to manage the active learning for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}/datasets/active"

    def get_active_set(self) -> MediaList:
        """
        Retrieve the active dataset for the project

        :return: MediaList containing the images and videoframes that are in the
            current active set for the project
        """
        result = self.session.get_rest_response(url=self.base_url, method="GET")
        active_set = MediaList([])
        for media_item in result["active_set"]:
            media_item.pop("dataset_id")
            if media_item["type"] == "image":
                active_set.append(MediaRESTConverter.from_dict(media_item, media_type=Image))
            elif media_item["type"] == "video":
                active_frames = media_item.pop("active_frames")
                video = MediaRESTConverter.from_dict(media_item, media_type=Video)
                frame_list = [video.get_frame(frame_index=i) for i in active_frames]
                active_set.extend(frame_list)
        return active_set
