# Deploying Geti models with OpenVINO Model Server (OVMS)
> Note: This feature is deprecated and is not recommended for use.

This README describes how to set up an OpenVINO Model Server for any Geti™
project. Please note that it is meant as an example only, and the configuration used
may not be optimal for a production environment.

Furthermore, this example shows how to run the OpenVINO Model Server and connect from it
**from the same system**. If you want to connect with a different client, additional
configuration on the server side may be required (opening the proper ports,
setting up authentication, etc.).

## Prerequisites
### Docker
To be able to follow the steps in this example, make sure you have docker installed on
your system. If you don't have Docker set up already, you can get it
[here](https://docs.docker.com/get-docker/).

### OVMS docker image
Fetch the latest OpenVINO Model Server docker image by executing
```shell
docker pull openvino/model_server:latest
```

## Firing up the OVMS container
Follow the steps below to run the OVMS container with your Geti™ trained
model(s):

    1. In your terminal, navigate to the directory containing the deployment you want to
        run in OVMS. It is the folder that contains this file (OVMS_README.md) and the
        `ovms_models` directory.

    2. Run the command:
        ```shell
        docker run -d --rm -v ${PWD}/ovms_models:/models -p 9000:9000 openvino/model_server:latest --port 9000 --log_level DEBUG --config_path /models/ovms_model_config.json --log_path /models/ovms_log.log
        ```

    3. The OpenVINO Model Server should now be running on your system, and listening
        for inference requests on port 9000.

## Running inference with Geti SDK and OVMS
The following python snippet can be used to run inference for your Geti™ project
on the OVMS instance that you just launched:
```python
from geti_sdk.deployment import Deployment

deployment = Deployment.from_folder(path_to_folder="deployment")

# Connect to the OVMS instance
deployment.load_inference_models(device="http://localhost:9000")

# Load example image
from geti_sdk.demos import EXAMPLE_IMAGE_PATH
import cv2

image = cv2.imread(EXAMPLE_IMAGE_PATH)
# Make sure to convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference on image
predictions = deployment.infer(image=image)

# Show inference result
from geti_sdk import Visualizer
visualizer = Visualizer()
result_image = visualizer.draw(image=image, annotation_scene=predictions)
visualizer.show_window(result_image)
```

The example uses a sample image, please make sure to replace it with your own.
