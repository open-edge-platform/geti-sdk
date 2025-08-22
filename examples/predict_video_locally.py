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

import argparse
import logging

from geti_sdk.demos import predict_video_from_deployment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict video on local machine.")
    parser.add_argument("video_path", type=str, help="File path to video")
    parser.add_argument(
        "deployment_path",
        type=str,
        help="Path to the folder containing the deployment data",
    )
    parser.add_argument(
        "--device",
        choices=["CPU", "GPU"],
        default="CPU",
        help="Device (CPU or GPU) to load the model to. Defaults to 'CPU'",
    )
    parser.add_argument(
        "--drop_audio",
        action="store_true",
        help="Option to drop audio. defaults to 'False'(preserving audio)",
    )
    parser.add_argument(
        "--log_level",
        choices=["warning", "info"],
        default="warning",
        help="Logging level. Defaults to 'warning'",
    )

    args = parser.parse_args()

    level_config = {"warning": logging.WARNING, "info": logging.INFO}
    log_level = level_config[args.log_level.lower()]
    logging.basicConfig(level=log_level)

    video_path = args.video_path
    deployment_path = args.deployment_path
    device = args.device
    preserve_audio = not args.drop_audio

    # Reconstruct video with overlaid predictions on local machine.
    predict_video_from_deployment(video_path, deployment_path, device=device, preserve_audio=preserve_audio)
