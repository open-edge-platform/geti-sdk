# Geti SDK Notebooks

This directory contains Jupyter notebooks that demonstrate how to use the Geti SDK for various computer vision workflows. These tutorials provide hands-on examples of the SDK's capabilities, from basic project creation to advanced deployment scenarios.

## Available Notebooks

### Core Functionality
- **[001_create_project.ipynb](001_create_project.ipynb)** - Learn how to create a new project from scratch and interact with it.
- **[002_create_project_from_dataset.ipynb](002_create_project_from_dataset.ipynb)** - Create a project from an existing dataset with images and annotations in various formats.
- **[003_export_and_import_projects.ipynb](003_export_and_import_projects.ipynb)** - Export projects to archives for backup or migration, and import them to recreate projects.
- **[004_train_project.ipynb](004_train_project.ipynb)** - Start training jobs, monitor progress, and retrieve trained models from completed jobs.
- **[005_manage_models.ipynb](005_manage_models.ipynb)** - View, compare, and manage different models within a project, including optimization workflows.

### Inference and Evaluation
- **[006_upload_and_predict_image.ipynb](006_upload_and_predict_image.ipynb)** - Upload images to existing projects and get predictions with visualization overlays.
- **[007_deploy_project.ipynb](007_deploy_project.ipynb)** - Create local deployments for running inference with OpenVINO on your own hardware.
- **[008_asynchronous_inference.ipynb](008_asynchronous_inference.ipynb)** - Perform efficient batch inference using asynchronous processing techniques.
- **[009_post_inference_hooks.ipynb](009_post_inference_hooks.ipynb)** - Implement custom post-processing workflows that run automatically after inference.

### Advanced Features
- **[010_modify_image.ipynb](010_modify_image.ipynb)** - Transform images (e.g., convert to grayscale) while preserving and reapplying annotations.
- **[011_change_project_configuration.ipynb](011_change_project_configuration.ipynb)** - View and modify project settings, training parameters, and task configurations.
- **[012_benchmarking_models.ipynb](012_benchmarking_models.ipynb)** - Measure and compare inference performance of different models on local hardware.
- **[013_download_and_upload_project.ipynb](013_download_and_upload_project.ipynb)** - Download complete projects (media, annotations, configuration) and recreate them on different servers.

### Use Cases
- **[101_simulate_low_light_product_inspection.ipynb](use_cases/101_simulate_low_light_product_inspection.ipynb)** - Simulate lighting condition changes and their effects on model predictions using MVTec AD dataset.
- **[102_from_zero_to_hero_9_steps.ipynb](use_cases/102_from_zero_to_hero_9_steps.ipynb)** - Complete end-to-end workflow from project creation to local inference deployment.
- **[103_parking_lot_train2deployment.ipynb](use_cases/103_parking_lot_train2deployment.ipynb)** - Build a smart car counting system for parking lots, including training, optimization, and deployment.

## Try the Notebooks

> [!IMPORTANT]
> All notebooks require access to a running Intel® Geti™ server.
> Make sure your server is accessible and your authentication is properly configured before running the notebooks.

### Setup the Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/openvinotoolkit/geti-sdk.git
   cd geti-sdk
   ```

2. **Install uv (if not already installed):**
   
   Follow the official [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

3. **Set up the environment with notebook dependencies:**
   ```bash
   uv sync --no-dev --extra notebooks
   ```

### Authentication Setup

Create a `.env` file in the `notebooks` directory with your Intel® Geti™ server details:

#### Option 1: Personal Access Token (Recommended)
```shell
# Geti instance details
HOST=https://your_server_hostname_or_ip_address
TOKEN=your_personal_access_token
```

To obtain a Personal Access Token:
1. Open Intel® Geti™ in your browser
2. Click the **User** menu (top right corner)
3. Select **Personal access token**
4. Follow the steps to create and copy your token

#### Option 2: Username and Password
```shell
# Geti instance details
HOST=https://your_server_hostname_or_ip_address
USERNAME=your_username
PASSWORD=your_password
```

> [!NOTE]
> If you specify both `TOKEN` and `USERNAME`/`PASSWORD`, the `TOKEN` will take precedence.


#### Optional: Disable SSL Certificate Verification
If your server doesn't have a valid SSL certificate (e.g., private network), add:
```shell
VERIFY_CERT=0
```

### Running the Notebooks

1. **Navigate to the notebooks directory:**
   ```bash
   cd notebooks
   ```

2. **Launch Jupyter Lab:**
   ```bash
   uv run --with jupyter jupyter lab
   ```

3. **Access the interface:**
   This will open your browser and take you to the Jupyter Lab interface where you can browse and run the notebooks.

4. **Start exploring:**
   Begin with `001_create_project.ipynb` for basic concepts, or jump to specific use cases that match your needs.

![Jupyter lab landing page](../docs/source/images/jupyter_lab_landing_page.png)