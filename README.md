<div align="center">

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
![Intel Geti](https://img.shields.io/badge/Intel%C2%AE%20Geti%E2%84%A2-2.12-blue?link=https%3A%2F%2Fgeti.intel.com%2F)
[![openvino](https://img.shields.io/badge/openvino-2025.2-purple)](https://github.com/openvinotoolkit/openvino)

![Pre-merge Tests Status](https://img.shields.io/github/actions/workflow/status/openvinotoolkit/geti-sdk/pre-merge-tests.yml?label=pre-merge%20tests&link=https%3A%2F%2Fgithub.com%2Fopenvinotoolkit%2Fgeti-sdk%2Factions%2Fworkflows%2Fpre-merge-tests.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8329/badge)](https://www.bestpractices.dev/projects/8329)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/open-edge-platform/geti-sdk/badge)](https://scorecard.dev/viewer/?uri=github.com/open-edge-platform/geti-sdk)

</div>

---

# Geti SDK

Geti SDK is a Python client for programmatically interacting with an [Intel® Geti™](https://github.com/open-edge-platform/geti) server via its [REST API](https://docs.geti.intel.com/docs/rest-api/openapi-specification).
With Geti SDK, you can automate and streamline computer vision workflows, making it easy to manage datasets, train models, and deploy solutions directly from your Python environment.

### What is Intel® Geti™?

[Intel® Geti™](https://github.com/open-edge-platform/geti) is an AI platform designed to help anyone build state-of-the-art computer vision models quickly and efficiently, even with minimal data.
It provides an end-to-end workflow for preparing, training, deploying, and running computer vision models at the edge. Geti™ supports the full AI model lifecycle, including dataset preparation, model training, and deployment of [OpenVINO™](https://docs.openvino.ai/)-optimized models.

### What can you do with Geti SDK?

With Geti SDK, you can:
- Create projects from annotated datasets or from scratch
- Upload and manage images, videos, and annotations
- Configure and update project and training settings
- Export and import datasets and projects, including models and configuration
- Deploy projects for local inference with OpenVINO
- Launch and monitor training, optimization, and evaluation workflows
- Run inference on images and videos
- And much more! See [Supported features](#supported-features) for more details.

### Tutorials and Examples

The ['Code examples'](#code-examples) sections below contains short snippets that demonstrate
how to perform several common tasks. This also shows how to configure the SDK to connect to your Intel® Geti™ server.

For more comprehensive examples, see the [Jupyter notebooks](https://github.com/openvinotoolkit/geti-sdk/tree/main/notebooks).
These tutorials demonstrate how to use the SDK for various computer vision tasks and workflows, from basic project creation 
to advanced inference scenarios.


## Install the SDK

Choose the installation method that best fits your use case:

### From PyPI

The easiest way to install the SDK is via [PyPI](https://pypi.org/project/geti-sdk).
This is the recommended method for most users who want to integrate Geti SDK into their own Python applications:

```bash
pip install geti-sdk
```

> [!IMPORTANT]
> Make sure to install a version of the SDK that matches your Geti server version. For example, if you're using Geti server version 2.12, install SDK version 2.12:
> ```bash
> pip install geti-sdk==2.12
> ```

#### Python and OS compatibility

Geti SDK currently supports the following operating systems and Python versions:

|             | Python <= 3.9 |    Python 3.10     | Python 3.11        | Python 3.12        | Python 3.13 |
|:------------|:-------------:|:------------------:|:------------------:|:------------------:|:-----------:|
| **Linux**   |      :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |     :x:     |
| **Windows** |      :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |     :x:     |
| **MacOS**   |      :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |     :x:     |

### From source

You can also choose to install the SDK from source by cloning the Git repository.
This option is useful for users who want to experiment with the SDK and notebooks,
or test the latest features before release, or for developers contributing to the project.

#### Install from a custom branch

Follow these steps to install the SDK from a specific branch or commit:

1. Clone the repository:
   ```bash
   git clone https://github.com/openvinotoolkit/geti-sdk.git
   cd geti-sdk
   ```

2. Checkout the desired branch or commit (e.g., for the 2.12 release):
   ```bash
   git checkout release-2.12
   ```
   Or use the develop branch for the latest changes:
   ```bash
   git checkout develop
   ```

3. Install the SDK:
   ```bash
   pip install .
   ```

#### Try the notebooks

To explore the SDK features through Jupyter notebooks, please see the detailed setup instructions in [notebooks/README.md](notebooks/README.md).

#### For developers

Developers who want to modify the SDK source code should follow the development setup instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## Code examples

The package provides a main class `Geti` that can be used for the following use cases:

### Connect to the Intel® Geti™ platform

To establish a connection between the SDK and the Intel® Geti™ platform, the `Geti` class needs to know the hostname or IP address for the server and requires authentication.

#### Personal Access Token (Recommended)

The recommended authentication method is the 'Personal Access Token'. To obtain a token:

1. Open the Intel® Geti™ user interface in your browser
2. Click on the `User` menu in the top right corner
3. Select `Personal access token` from the dropdown menu
4. Follow the steps to create a token and copy the token value

![Personal access token menu](docs/source/images/personal_access_token.png)

```python
from geti_sdk import Geti

geti = Geti(
    host="https://your_server_hostname_or_ip_address",
    token="your_personal_access_token"
)
```

#### User Credentials

It is also possible to authenticate using a username and password:

```python
from geti_sdk import Geti

geti = Geti(
    host="https://your_server_hostname_or_ip_address",
    username="your_username",
    password="your_password"
)
```

> [!NOTE]
> By default, the SDK verifies SSL certificates. To disable certificate validation (only in secure environments),
pass the `verify_certificate=False` argument to the `Geti` constructor.

### Manage projects

#### Create a new project

```python
from geti_sdk import Geti
from geti_sdk.rest_clients import ProjectClient

geti = Geti(host="https://your_server", token="your_token")
project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)

# Create a detection project
project = project_client.create_project(
    project_name="My Detection Project",
    project_type="detection",
    labels=[["person", "car", "bicycle"]]
)
```

#### Get an existing project

```python
from geti_sdk.rest_clients import ProjectClient

project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)

# Get project by name
project = project_client.get_project_by_name("My Detection Project")

# List all projects
projects = project_client.list_projects()
```

### Upload and annotate media

#### Upload an image

```python
import cv2
from geti_sdk.rest_clients import ImageClient, AnnotationClient

# Set up clients
image_client = ImageClient(session=geti.session, workspace_id=geti.workspace_id, project=project)
annotation_client = AnnotationClient(session=geti.session, workspace_id=geti.workspace_id, project=project)

# Upload image
image = cv2.imread("path/to/your/image.jpg")
uploaded_image = image_client.upload_image(image)
```

#### Annotate an image

```python
from geti_sdk.data_models import Rectangle, Annotation

# Create a bounding box annotation
bbox = Rectangle(x=100, y=100, width=200, height=150)
annotation = Annotation(
    shape=bbox,
    labels=[project.get_trainable_tasks()[0].labels[0]]  # Use first label
)

# Upload annotation
annotation_client.upload_annotation_for_image(uploaded_image, annotation)
```

#### List media

```python
from geti_sdk.rest_clients import ImageClient

image_client = ImageClient(session=geti.session, workspace_id=geti.workspace_id, project=project)

# Get all images in a project
images = image_client.get_all_images()
print(f"Found {len(images)} images in the project")

# Get images from specific dataset
dataset = project.datasets[0]  # Get first dataset
images_in_dataset = image_client.get_images_in_dataset(dataset)
```

### Train a project

```python
from geti_sdk.rest_clients import TrainingClient
import time

training_client = TrainingClient(session=geti.session, workspace_id=geti.workspace_id, project=project)

# Start training
job = training_client.train_project()
print(f"Training job started with ID: {job.id}")

# Monitor training progress
while not job.is_finished:
    time.sleep(30)  # Wait 30 seconds
    job = training_client.get_job_by_id(job.id)
    print(f"Training status: {job.status}")

print("Training completed!")
```

### Run inference on an image

```python
from geti_sdk.rest_clients import ImageClient, PredictionClient

image_client = ImageClient(session=geti.session, workspace_id=geti.workspace_id, project=project)
prediction_client = PredictionClient(session=geti.session, workspace_id=geti.workspace_id, project=project)

# Upload image and get prediction
image = cv2.imread('path/to/test_image.jpg')
uploaded_image = image_client.upload_image(image)
prediction = prediction_client.get_image_prediction(uploaded_image)
```

### Import/export

**Export and import a project**

```python
from geti_sdk.import_export import GetiIE

# Set up the import/export client
geti_ie = GetiIE(workspace_id=geti.workspace_id, session=geti.session, project_client=project_client)

# Get the project to export
project = project_client.get_project_by_name("My Detection Project")

# Export project as zip archive
geti_ie.export_project(
    project_id=project.id,
    filepath="./my_project_export.zip",
    include_models="all"  # Options: 'all', 'none', 'latest_active'
)

# Import project from zip archive
imported_project = geti_ie.import_project(
    filepath="./my_project_export.zip",
    project_name="Imported Project"  # Optional: specify new name
)
```

**Export and import a dataset**

```python
from geti_sdk.import_export import GetiIE
from geti_sdk.data_models.enums import DatasetFormat

# Set up the import/export client
geti_ie = GetiIE(session=geti.session, workspace_id=geti.workspace_id, project_client=project_client)

# Export dataset in Datumaro format
dataset = project.datasets[0]  # Get first dataset
geti_ie.export_dataset(
    project=project,
    dataset=dataset,
    filepath="./dataset_export.zip",
    export_format=DatasetFormat.DATUMARO,
    include_unannotated_media=False
)

# Import dataset as new project
imported_project = geti_ie.import_dataset_as_new_project(
    filepath="./dataset_export.zip",
    project_name="Project from Dataset",
    project_type="detection"
)
```


## Supported features

Geti SDK supports most of the operations that are exposed via the [Geti REST API](https://docs.geti.intel.com/docs/rest-api/openapi-specification),
although some advanced features may not be available yet due to technical and security reasons.

- [x] **Manage projects and their configuration** - Create, delete, and reconfigure projects of any type, including multi-task pipelines
- [x] **Upload media** - Upload images and videos with various formats and resolutions
- [x] **Annotate media** - Create annotations for images and video frames
- [x] **Train, optimize and evaluate models** - Launch training jobs, trigger post-training optimization (quantization) and evaluate models on custom datasets
- [x] **Monitor long-running workflows** - Track the status and progress of training, optimization, and evaluation jobs
- [x] **Generate predictions with a trained model** - Upload media and get predictions, with support for both single images and batch processing
- [x] **Active learning** - Get suggestions for the most informative samples to annotate next
- [x] **Get statistics about datasets and models** - Retrieve comprehensive statistics and metrics for datasets and models
- [x] **Deploy and benchmark models locally** - Export OpenVINO inference models, run full pipeline inference on local machines, and measure inference throughput on your hardware configurations
- [x] **Download and upload datasets** - Export datasets to archives and import them to create new projects
- [x] **Download and upload full projects** - Create complete backups of projects, including datasets, models and configurations, and restore them
- [ ] **Upload trained models** - Intel® Geti™ does not allow to import external models
- [ ] **Import datasets to existing projects** - currently, this feature is only available through the Intel® Geti™ UI and API
- [ ] **Manage users and roles** - currently, this feature is only available through the Intel® Geti™ UI and API

Are you looking for a specific feature that is not listed here?
Please check if it is implemented by one of the clients in the [rest_clients](geti_sdk/rest_clients) module,
else feel free to open an issue or contribute a pull request (see the [CONTRIBUTING](CONTRIBUTING.md) guidelines).