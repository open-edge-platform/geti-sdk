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
import csv
import itertools
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk import Geti
from geti_sdk.data_models import (
    Image,
    Model,
    OptimizedModel,
    Performance,
    Project,
    Video,
)
from geti_sdk.deployment import Deployment
from geti_sdk.prediction_visualization.visualizer import Visualizer
from geti_sdk.rest_clients import ImageClient, ModelClient, TrainingClient, VideoClient
from geti_sdk.rest_clients.prediction_client import PredictionClient
from geti_sdk.utils.plot_helpers import (
    concat_prediction_results,
    pad_image_and_put_caption,
)

from .utils import get_system_info, load_benchmark_media, suppress_log_output


class Benchmarker:
    """
    Initialize and manage benchmarking experiments to measure model throughput on
    different hardware.
    """

    def __init__(
        self,
        geti: Geti,
        project: Project,
        precision_levels: Sequence[str] | None = None,
        models: Sequence[Model] | None = None,
        algorithms: Sequence[str] | None = None,
        benchmark_images: Sequence[Image | np.ndarray | os.PathLike] | None = None,
        benchmark_video: Video | os.PathLike | None = None,
    ):
        """
        Manage benchmarking experiments to measure inference model throughput on
        different hardware. It allows for quick and easy comparison of inference
        framerates for different model architectures and precision levels for the
        specified project.

        The Benchmarker will fetch models from Intel Geti and measure the
        throughput for local inference for these models. It requires an existing Geti
        Project to work. The project does not need to be trained yet, but must have
        sufficient annotations to be able to start training.

        The benchmark can be run on any target device that is supported by OpenVINO.
        Specifying multiple algorithms and precision levels to include in the
        benchmark allows for easy comparison of the performance for these models.

        NOTE: For task chain projects, passing `models` or `algorithms` directly in the
        constructor is not supported. The Benchmarker provides dedicated
        methods to specify which models or algorithms to use in this case, which can
        be called after initialization.

        :param geti: Geti instance on which the project to use for benchmarking lives
        :param project: Project to use for the benchmarking. The
            project must exist on the specified Geti instance
        :param precision_levels: List of model precision levels to run the
            benchmarking for. Throughput will be measured for each precision level
            in this list. Valid options are ["FP32", "FP16", "INT8"]. Defaults to:
            ["FP32", "FP16"]
        :param models: Optional list of base models to use in the benchmarking.
            Note that this should not be optimized models: The Benchmarker will fetch
            optimized models in the specified `precision_levels` for the base
            models specified here
        :param algorithms: List of algorithm names for which the benchmarking should
            be performed. If no models are trained for all algorithms in the list,
            the Benchmarker will trigger training for those algorithms and wait for
            the training jobs to complete. If neither `models` or `algorithms` are
            specified, the Benchmarker uses the current active model(s) in the project.
        :param benchmark_images: Optional list of images to run the benchmarks on. Can
            be passed as Geti Images, numpy images or filepaths to image files on disk.
            NOTE: If no images are specified, the benchmark will use a random image
            or video frame from the project
        :param benchmark_video: Optional video to use for the benchmarking. Can be
            passed either as a Geti Video object, or as a filepath to a video file on
            disk.
            NOTE: Either `benchmark_video` or `benchmark_images` can be specified, but
            not both. If neither images nor video is passed, the Benchmarker will
            pick a random image or video frame from the project to run the benchmark
            on.
        """
        self.geti = geti
        # Update project object to get the latest project details
        self.project = self.geti.get_project(project_id=project.id)
        logging.info(f"Setting up Benchmarker for Intel® Geti™ project `{self.project.name}`.")
        self._is_single_task = len(self.project.get_trainable_tasks()) == 1
        if precision_levels is None:
            precision_levels = ["FP32", "FP16"]
        self.precision_levels = precision_levels
        self.model_client = ModelClient(session=geti.session, workspace_id=geti.workspace_id, project=self.project)
        self.training_client = TrainingClient(
            session=geti.session, workspace_id=geti.workspace_id, project=self.project
        )

        self._models: list[Model] | None = None
        self._task_chain_models: list[list[Model]] | None = None
        self._algorithms: list[str] | None = None
        self._task_chain_algorithms: list[list[str]] | None = None
        self._optimized_models: list[OptimizedModel] | None = None
        self._task_chain_optimized_models: list[list[OptimizedModel]] | None = None
        self._deployment_folders: list[os.PathLike] = []

        if not self._is_single_task and (models is not None or algorithms is not None):
            raise ValueError(
                "You have specified a task-chain project to be used for benchmarking. "
                "This does not allow setting `models` or `algorithms` via the "
                "Benchmark constructor. Please specify which models or algorithms "
                "should be used in the benchmark using the methods "
                "`Benchmarker.set_task_chain_models()` or "
                "`Benchmarker.set_task_chain_algorithms()`. These methods "
                "can be used after initialization of the Benchmarker. "
                "Please refer to the method documentation for further guidance."
            )

        if models is None and algorithms is None:
            if self._is_single_task:
                logging.info("No `models` or `algorithms` were specified, using current active models in the project.")
            else:
                logging.info(
                    "Models or algorithms to benchmark for a task chain project can "
                    "be specified by calling the `Benchmarker.set_task_chain_models()`"
                    " or `Benchmarker.set_task_chain_algorithms()` methods. If these "
                    "methods are not called, the benchmark will be set up using the "
                    "current active models."
                )
            models = self.model_client.get_all_active_models()
            if len(models) == 0:
                raise ValueError(
                    "No trained models were found in the project, please either "
                    "train a model first or specify an algorithm to train."
                )
            algorithms: list[str] = []
            for model in models:
                task = self.model_client.get_task_for_model(model=model)
                if self._is_single_task:
                    logging.info(f"Found active model `{model.name}` for task `{task.title}`")
                algorithms.append(model.architecture)
            if self._is_single_task:
                self._models = models
                self._algorithms = algorithms
            else:
                self._task_chain_models = [[model] for model in models]
                self._task_chain_algorithms = [[algo] for algo in algorithms]
            self._are_models_specified = True

        elif models is not None and algorithms is not None:
            raise ValueError("Either `models` or `algorithms` could be specified, but not both.")
        elif models is not None and algorithms is None:
            self._models = models
            self._are_models_specified = True
            self._algorithms = [model.architecture for model in models]
        elif models is None and algorithms is not None:
            self._models = None
            self._are_models_specified = False
            self._algorithms = algorithms

        self.images = None
        self.video = None
        if benchmark_video is not None and benchmark_images is not None:
            raise ValueError("Please specify either `benchmark_video` or `benchmark_images`, but not both.")
        if benchmark_images is not None:
            self.images = benchmark_images
        elif benchmark_video is not None:
            self.video = benchmark_video
        else:
            image_client = ImageClient(
                session=geti.session,
                workspace_id=geti.workspace_id,
                project=self.project,
            )
            images = image_client.get_all_images()
            if len(images) > 0:
                self.images = [images[0]]
            else:
                video_client = VideoClient(
                    session=geti.session,
                    workspace_id=geti.workspace_id,
                    project=self.project,
                )
                videos = video_client.get_all_videos()
                if len(videos) > 0:
                    self.video = videos[0]
                else:
                    raise ValueError(
                        "No benchmark images or video was specified, and no media "
                        "could be found in the project. Unable to proceed, please "
                        "specify either images or a video to run the benchmark on."
                    )

    def set_task_chain_models(self, models_task_1: Sequence[Model], models_task_2: Sequence[Model]):
        """
        Set the models to be used in the benchmark for a task-chain project. The
        benchmarking will run for all possible combinations of models for task 1
        and task 2.

        :param models_task_1: Models to use for task #1
        :param models_task_2: Models to use for task #2
        """
        if self._is_single_task:
            logging.warning(
                "Method `set_task_chain_models` was called for a non-task-chain project. This has no effect."
            )
            return
        self._task_chain_models = [list(pair) for pair in itertools.product(models_task_1, models_task_2)]
        self._are_models_specified = True
        self._task_chain_algorithms = [[model.architecture for model in pair] for pair in self._task_chain_models]
        logging.info(
            f"Task chain models set. Found a total of {len(self._task_chain_models)} possible combinations to benchmark"
        )

    def set_task_chain_algorithms(self, algorithms_task_1: Sequence[str], algorithms_task_2: Sequence[str]):
        """
        Set the algorithms to be used in the benchmark for a task-chain project. The
        benchmarking will run for all possible combinations of algorithms for task 1
        and task 2.

        Note that upon benchmark initialization, the Benchmarker will check if
        trained models are available for all algorithms specified

        :param algorithms_task_1: Algorithms to use for task #1
        :param algorithms_task_2: Algorithms to use for task #2
        """
        if self._is_single_task:
            logging.warning(
                "Method `set_task_chain_algorithms` was called for a non-task-chain project. This has no effect."
            )
            return
        self._task_chain_algorithms = [list(pair) for pair in itertools.product(algorithms_task_1, algorithms_task_2)]
        self._are_models_specified = False
        self._task_chain_models = None
        logging.info(
            f"Task chain algorithms set. Found a total of "
            f"{len(self._task_chain_algorithms)} possible algorithm combinations "
            f"to benchmark"
        )

    @property
    def models(self) -> list[Model] | list[list[Model]]:
        """
        Return the models to be used in the benchmark.
        """
        if self._are_models_specified:
            if self._is_single_task:
                return self._models
            return self._task_chain_models
        raise ValueError("Unable to access models, no benchmark models have been specified yet.")

    @property
    def algorithms(self) -> list[str] | list[list[str]]:
        """
        Return the algorithm names to be used in the benchmark
        """
        if self._is_single_task:
            return self._algorithms
        return self._task_chain_algorithms

    @property
    def optimized_models(
        self,
    ) -> list[OptimizedModel] | list[list[OptimizedModel]]:
        """
        Return the optimized models to be used in deployments for the benchmark
        """
        if self._is_single_task and self._optimized_models is not None:
            return self._optimized_models
        if not self._is_single_task and self._task_chain_optimized_models is not None:
            return self._task_chain_optimized_models
        raise ValueError(
            "Optimized models have not been assigned yet. Please initialize the "
            "benchmarker first using the `initialize_benchmark` method."
        )

    def _train_model_for_algorithm(self, task_index: int, algorithm_name: str) -> Model:
        """
        Train a model for the specified task and algorithm

        :param task_index: Index of the task for which to train the model
        :param algorithm_name: Name of the algorithm to use
        :return: Model object, representing the trained model
        """
        algos = self.training_client.get_algorithms_for_task(task_index)
        algo = algos.get_by_name(algorithm_name)
        job = self.training_client.train_task(task_index, algorithm=algo)
        job = self.training_client.monitor_job(job=job)
        return self.model_client.get_model_for_job(job)

    def _optimize_model_for_algorithm(self, model: Model, precision: str = "INT8") -> OptimizedModel:
        """
        Optimize a `model` with the given `precision`.

        :param model: Base model to optimize
        :param precision: Precision to use for the optimization
        :return: OptimizedModel object, representing the optimized model
        """
        job = self.model_client.optimize_model(model=model)
        self.model_client.monitor_job(job)
        updated_model = self.model_client.update_model_detail(model)
        return updated_model.get_optimized_model(precision=precision, optimization_type="openvino")

    def prepare_benchmark(self, working_directory: os.PathLike = "."):
        """
        Prepare the benchmarking experiments. This involves:

            1. Ensuring that all required models are available, i.e. that all
                specified algorithms have a trained model in the Geti project. If
                not, training jobs will be started and awaited.
            2. Ensuring that for each model, optimized models with the required
                quantization level are available. If not, optimization jobs will
                be started and awaited.
            3. Creating and downloading deployments for all models to benchmark.

        :param working_directory: Output directory to which the deployments for the
            benchmark will be saved.
        """
        logging.info("Preparing benchmark experiments.")
        # First, check if model training is required. This happens when
        # algorithms are specified, but not all algorithms may have a model
        # trained already.
        if not self._are_models_specified:
            if self._is_single_task:
                logging.info(
                    f"Checking model availability for {len(self.algorithms)} different algorithms: {self.algorithms}."
                )
                models: list[Model] = []
                for algorithm_name in self.algorithms:
                    model = self.model_client.get_latest_model_by_algo_name(algorithm_name)
                    if model is None:
                        logging.info(
                            f"No model found in project for algorithm {algorithm_name}, requesting model training"
                        )
                        model = self._train_model_for_algorithm(task_index=0, algorithm_name=algorithm_name)
                    models.append(model)
                self._models = models
            else:
                logging.info(f"Checking model availability for {len(self.algorithms)} different pairs of algorithms.")
                models: list[list[Model]] = []
                for algo_pair in self.algorithms:
                    model_pair: list[Model] = []
                    for task_index, algorithm_name in enumerate(algo_pair):
                        model = self.model_client.get_latest_model_by_algo_name(algorithm_name)
                        if model is None:
                            model = self._train_model_for_algorithm(
                                task_index=task_index, algorithm_name=algorithm_name
                            )
                        model_pair.append(model)
                    models.append(model_pair)
                self._task_chain_models = models
            self._are_models_specified = True
        logging.info("All required base models are available")
        # Then, check if model optimization is required. Create a list of
        # optimized models for which deployments should be created.
        if self._is_single_task:
            logging.info(
                f"Checking optimized model availability for {len(self.models)} "
                f"different models and quantization levels: {self.precision_levels}"
            )
            optimized_models: list[OptimizedModel] = []
            for model in self.models:
                for precision in self.precision_levels:
                    opt_model = model.get_optimized_model(precision=precision, optimization_type="openvino")
                    if opt_model is None:
                        logging.info(
                            f"No optimized model with quantization level {precision} "
                            f"found for model {model.name}, requesting model "
                            f"optimization."
                        )
                        opt_model = self._optimize_model_for_algorithm(model=model, precision=precision)
                    optimized_models.append(opt_model)
            self._optimized_models = optimized_models
        else:
            logging.info(
                f"Checking optimized model availability for {len(self.models)} "
                f"different model pairs and quantization levels: "
                f"{self.precision_levels}"
            )
            optimized_models: list[list[OptimizedModel]] = []
            for model_pair in self.models:
                for precision in self.precision_levels:
                    opt_model_pair: list[OptimizedModel] = []
                    for model in model_pair:
                        # Update model info, to take into account any
                        # optimizations made during loop execution
                        model = self.model_client.update_model_detail(model=model)
                        opt_model = model.get_optimized_model(precision=precision, optimization_type="openvino")
                        if opt_model is None:
                            logging.info(
                                f"No optimized model with quantization level {precision} "
                                f"found for model {model.name}, requesting model "
                                f"optimization."
                            )
                            opt_model = self._optimize_model_for_algorithm(model=model, precision=precision)
                        opt_model_pair.append(opt_model)
                    optimized_models.append(opt_model_pair)
            self._task_chain_optimized_models = optimized_models
        logging.info("All required optimized models are available")

        # Next up, create deployments for all optimized models
        os.makedirs(working_directory, exist_ok=True)
        logging.info(
            f"Creating {len(self.optimized_models)} deployments to benchmark. Saving "
            f"deployment data to folder: `{working_directory}`"
        )
        with logging_redirect_tqdm(tqdm_class=tqdm):
            for index, opt_models in tqdm(
                enumerate(self.optimized_models),
                total=len(self.optimized_models),
                desc="Creating deployments",
            ):
                if isinstance(opt_models, OptimizedModel):
                    opt_models = [opt_models]
                output_folder = os.path.join(working_directory, f"deployment_{index}")
                with suppress_log_output():
                    self.geti.deploy_project(
                        project=self.project,
                        output_folder=output_folder,
                        models=opt_models,
                    )
                    self._deployment_folders.append(output_folder)
        logging.info("Deployments created. Benchmark initialization complete.")

    def initialize_from_folder(self, target_folder: os.PathLike = "."):
        """
        Initialize the Benchmarker from a folder containing deployments. This method
        checks that any directory inside the `target_folder` contains a valid
        deployment for the project assigned to this Benchmarker.

        :param target_folder: Directory containing the model deployments that should
            be used in the Benchmarking.
        """
        logging.info(f"Initializing Benchmarker from folder `{target_folder}`")
        if len(self._deployment_folders) != 0:
            raise ValueError("The Benchmarker appears to be already initialized. Unable to continue.")
        for obj in os.listdir(target_folder):
            object_path = os.path.join(target_folder, obj)
            if os.path.isfile(object_path):
                continue
            try:
                deployment = Deployment.from_folder(path_to_folder=object_path)
            except ValueError:
                logging.info(f"Directory {object_path} does not contain a valid Deployment, skipping...")
                continue
            if deployment.project.name != self.project.name:
                raise ValueError(
                    f"Deployment at path `{object_path}` was created for project "
                    f"{deployment.project.name}, yet the Benchmarker was set up for "
                    f"project {self.project.name}. Unable to initialize. Please use "
                    f"only deployments created for project `{self.project.name}`"
                )
            self._deployment_folders.append(object_path)

        if len(self._deployment_folders) != 0:
            logging.info(
                f"Benchmarker was initialized from folder. {len(self._deployment_folders)} "
                f"valid deployment folders were found."
            )

    def run_throughput_benchmark(
        self,
        working_directory: os.PathLike = ".",
        results_filename: str = "results",
        target_device: str = "CPU",
        frames: int = 200,
        repeats: int = 3,
    ) -> list[dict[str, str]]:
        """
        Run the benchmark experiment.

        :param working_directory: Directory in which the deployments that should be
            benchmarked are stored. All output will be saved to this directory.
        :param results_filename: Name of the file to which the results will be saved.
            File extension should not be included, the results will always be saved as
            a `.csv` file. Defaults to `results.csv`. The results file will be created
            within the `working_directory`
        :param target_device: Device to run the inference models on, for example "CPU"
            or "GPU". Defaults to "CPU".
        :param frames: Number of frames/images to infer in order to calculate
            fps
        :param repeats: Number of times to repeat the benchmark runs. FPS will be
            averaged over the runs.
        """
        if len(self._deployment_folders) == 0:
            raise ValueError(
                "Benchmarker does not contain any deployments to benchmark yet! Please "
                "prepare the deployments first using either the "
                "`Benchmarker.prepare_benchmark()` or "
                "`Benchmarker.initialize_from_folder()` methods."
            )
        logging.info("Starting throughput benchmark experiments.")

        logging.info(f"The Benchmarker will run for {len(self._deployment_folders)} deployments")

        logging.info("Loading benchmark media")
        benchmark_frames = load_benchmark_media(
            session=self.geti.session,
            images=self.images,
            video=self.video,
            frames=frames,
        )

        results_file = os.path.join(working_directory, f"{results_filename}.csv")
        logging.info(f"Writing results to `{results_file}`")

        logging.info(f"Benchmarking inference rate for synchronous inference on {frames} frames with {repeats} repeats")
        with logging_redirect_tqdm(tqdm_class=tqdm):
            results: list[dict[str, str]] = []
            for deployment_index, deployment_folder in enumerate(tqdm(self._deployment_folders, desc="Benchmarking")):
                success = True
                deployment = Deployment.from_folder(deployment_folder)
                try:
                    with suppress_log_output():
                        deployment.load_inference_models(device=target_device)
                except Exception as e:
                    success = False
                    logging.info(
                        f"Failed to load inference models for deployment at path: "
                        f"`{deployment_folder}`, with error: {e}. Marking benchmark "
                        f"run for the deployment as failed"
                    )

                if success:
                    try:
                        # Warm-up for the model
                        deployment.infer(benchmark_frames[0])

                        # Estimate time to completion
                        t_single_start = time.time()
                        deployment.infer(benchmark_frames[0])
                        single_inf_time = time.time() - t_single_start
                        logging.info(
                            f"Inference model(s) for deployment `{deployment_folder}` "
                            f"loaded. Starting benchmark run. Estimated time required: "
                            f"{repeats * frames * single_inf_time:.0f} seconds"
                        )
                    except Exception as e:
                        success = False
                        logging.info(
                            f"Failed to run inference on frame number 0. Marking "
                            f"benchmark run for deployment `{deployment_folder}` as "
                            f"failed. Inference failed with error: `{e}`"
                        )
                    t_start = time.time()
                    for run in range(repeats):
                        for frame in benchmark_frames:
                            try:
                                deployment.infer(frame)
                            except Exception as e:
                                success = False
                                logging.info(
                                    f"Failed to run inference on frame number "
                                    f"{benchmark_frames.index(frame)}. Marking "
                                    f"benchmark run for deployment "
                                    f"`{deployment_folder}` as failed. Inference "
                                    f"failed with error: `{e}`"
                                )
                    t_elapsed = time.time() - t_start
                    total_frames = frames * repeats
                    fps = t_elapsed and total_frames / t_elapsed or 0
                else:
                    fps = 0

                model_scores = []
                for om in deployment.models:
                    if isinstance(om.performance, Performance):
                        score = om.performance.score
                    elif isinstance(om.performance, dict):
                        score = om.performance.get("score", -1)
                    else:
                        score = -1
                    model_scores.append(score)

                # Update result list
                result_row: dict[str, str] = {}
                result_row["name"] = f"Deployment {deployment_index}"
                result_row["project_name"] = self.project.name
                result_row["target_device"] = target_device
                result_row["task 1"] = self.project.get_trainable_tasks()[0].title
                result_row["model 1"] = deployment.models[0].name
                result_row["model 1 score"] = f"{model_scores[0]:.2f}"
                if not self._is_single_task:
                    result_row["task 2"] = self.project.get_trainable_tasks()[1].title
                    result_row["model 2"] = deployment.models[1].name
                    result_row["model 2 score"] = f"{model_scores[1]:.2f}"
                result_row["success"] = str(int(success))
                result_row["fps"] = f"{fps:.2f}"
                result_row["total frames"] = f"{frames * repeats}"
                result_row["source"] = deployment_folder
                result_row.update(get_system_info(device=target_device))

                # Write results to file
                fieldnames = list(result_row.keys())
                if not results:  # First row
                    with open(results_file, "w", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow(result_row)
                else:  # Append
                    with open(results_file, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result_row)
                results.append(result_row)

        return results

    def _predict_using_active_model(
        self,
        numpy_image: np.ndarray,
    ) -> dict[str, Any]:
        """
        Predict on platform using the active model.

        :param numpy_image: Numpy array containing the image to be predicted on.
        :return: Dictionary containing the prediction results.
        """
        # Predict on the Platform
        prediction_client = PredictionClient(
            session=self.geti.session,
            workspace_id=self.geti.workspace_id,
            project=self.project,
        )
        platform_prediction = prediction_client.predict_image(numpy_image)

        # load active models info
        active_models = self.model_client.get_all_active_models()
        result: dict[str, Any] = {}
        result["prediction"] = platform_prediction
        result["run_name"] = "On-Platform Prediction"
        result["model_1"] = active_models[0].name + " " + active_models[0].precision[0]
        result["model_1_score"] = active_models[0].performance.score
        if not self._is_single_task:
            result["model_2"] = active_models[1].name + " " + active_models[1].precision[0]
            result["model_2_score"] = active_models[1].performance.score
        return result

    def _add_header_to_comparison(self, comparison_image: np.ndarray, target_device: str) -> np.ndarray:
        """
        Add a header to the comparison image.

        :param comparison_image: Comparison image to add the header to.
        :return: Comparison image with header.
        """
        # Calculate text and image padding size
        text_scale = round(comparison_image.shape[1] / 1280, 1)
        thickness = int(text_scale / 1.4)
        (_, label_height), baseline = cv2.getTextSize("Test caption", cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
        top_padding_per_line = label_height + baseline
        # Prepare image captions
        device_info = get_system_info(device=target_device)["device_info"]
        caption_lines = [
            "Inference results comparison",
            f"Project: {self.project.name}",
            (
                f"Task: {self.project.get_trainable_tasks()[0].title}"
                + ("" if self._is_single_task else f" -> {self.project.get_trainable_tasks()[1].title}")
            ),
            f"Device info: {device_info}",
        ]
        # Pad the image
        padded_image = cv2.copyMakeBorder(
            comparison_image,
            top=2 * baseline + top_padding_per_line * len(caption_lines),
            bottom=0,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        # Put text
        for line_number, text_line in enumerate(caption_lines):
            cv2.putText(
                padded_image,
                text_line,
                (10, top_padding_per_line * (line_number + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),
                thickness,
            )
        return padded_image

    def compare_predictions(
        self,
        working_directory: os.PathLike = ".",
        saved_image_name: str = "comparison",
        target_device: str = "CPU",
        image: np.ndarray | str | os.PathLike | None = None,
        include_online_prediction_for_active_model: bool = True,  # the name is not finalized
        throughput_benchmark_results: list[dict[str, str]] | os.PathLike | None = None,
    ) -> np.ndarray:
        """
        TODO blank image if not success

        Perform visual comparison of predictions from different deployments.

        :param working_directory: Directory in which the deployments that should be
            benchmarked are stored. All output will be saved to this directory.
        :param saved_image_name: Name of the file to which the results will be saved.
            File extension should not be included, the results will always be saved as
            a `.jpg` file. Defaults to `comparison.jpg`. The results file will be created
            within the `working_directory`
        :param target_device: Device to run the inference models on, for example "CPU"
            or "GPU". Defaults to "CPU".
        :param image: Image to use for comparison. If no image is passed, the first
            image in the `images` list will be used.
        :param include_online_prediction_for_active_model: Flag to include prediction
            from the active model on the platform side.
        :param throughput_benchmark_results: Results from a throughput benchmark run. If
            this is passed, the captions for the images will contain the benchmark results.
        :return: Image containing visual comparison in form of a NumPy array.
        """
        if len(self._deployment_folders) == 0:
            raise ValueError(
                "Benchmarker does not contain any deployments to benchmark yet! Please "
                "prepare the deployments first using either the "
                "`Benchmarker.prepare_benchmark()` or "
                "`Benchmarker.initialize_from_folder()` methods."
            )
        logging.info("Starting collecting predictions for visual comparison.")

        logging.info(f"The Benchmarker will run for {len(self._deployment_folders)} deployments")

        logging.info("Loading benchmark media")
        if isinstance(image, np.ndarray):
            pass
        elif image is None:
            image = load_benchmark_media(
                session=self.geti.session,
                images=self.images,
                video=self.video,
                frames=1,
            )[0]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, str | os.PathLike):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f"Invalid image type: {type(image)}.")

        saved_image_path = os.path.join(working_directory, f"{saved_image_name}.jpg")
        logging.info(f"Saving visual comparison to `{saved_image_path}`")

        # Check the benchmark results
        if isinstance(throughput_benchmark_results, os.PathLike | str):
            with open(throughput_benchmark_results) as results_file:
                throughput_benchmark_results = list(csv.DictReader(results_file))

        visualizer = Visualizer()

        # Performe inferece
        with logging_redirect_tqdm(tqdm_class=tqdm):
            results: list[list[np.ndarray]] = []
            model_name_to_row: dict[str, int] = {}
            for deployment_index, deployment_folder in enumerate(
                tqdm(self._deployment_folders, desc="Collecting predictions")
            ):
                success = True
                deployment = Deployment.from_folder(deployment_folder)
                try:
                    with suppress_log_output():
                        deployment.load_inference_models(device=target_device)
                except Exception as e:
                    success = False
                    logging.info(
                        f"Failed to load inference models for deployment at path: "
                        f"`{deployment_folder}`, with error: {e}. Marking benchmark "
                        f"run for the deployment as failed"
                    )

                if success:
                    try:
                        prediction = deployment.infer(image)
                    except Exception as e:
                        success = False
                        logging.info(
                            f"Failed to run inference on the image. Marking "
                            f"benchmark run for deployment `{deployment_folder}` as "
                            f"failed. Inference failed with error: `{e}`"
                        )
                if success:
                    image_with_prediction = visualizer.draw(image, prediction)
                    image_with_prediction = cv2.cvtColor(image_with_prediction, cv2.COLOR_BGR2RGB)
                else:
                    # Replace the image with an empty one in case of no prediction
                    image_with_prediction = np.zeros_like(image)
                    image_with_prediction = cv2.putText(
                        image_with_prediction,
                        "Failed to run inference on the image",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )
                # Save model scores
                model_scores = []
                for om in deployment.models:
                    if isinstance(om.performance, Performance):
                        score = om.performance.score
                    elif isinstance(om.performance, dict):
                        score = om.performance.get("score", -1)
                    else:
                        score = -1
                    model_scores.append(score)

                model_info = {
                    "run_name": f"Deployment {deployment_index}",
                    "model_1": deployment.models[0].name,
                    "model_1_score": model_scores[0],
                    "fps": (
                        None
                        if throughput_benchmark_results is None
                        else throughput_benchmark_results[deployment_index]["fps"]
                    ),
                }
                if not self._is_single_task:
                    model_info.update(
                        {
                            "model_2": deployment.models[1].name,
                            "model_2_score": model_scores[1],
                        }
                    )
                # Pad the image and put captions on it
                image_with_prediction = pad_image_and_put_caption(image=image_with_prediction, **model_info)

                # Image is ready, now we add it to the results array
                # Determine result's position on the grid
                model_1_name = deployment.models[0].name.split(" ")[0]
                if model_1_name not in model_name_to_row:
                    model_name_to_row[model_1_name] = len(results)
                    results.append([])
                row_n = model_name_to_row[model_1_name]
                results[row_n].append(image_with_prediction)

        if include_online_prediction_for_active_model:
            logging.info("Predicting on the platform using the active model")
            online_prediction_result = self._predict_using_active_model(image)
            image_with_prediction = visualizer.draw(image, online_prediction_result["prediction"])
            image_with_prediction = cv2.cvtColor(image_with_prediction, cv2.COLOR_BGR2RGB)

            del online_prediction_result["prediction"]
            image_with_prediction = pad_image_and_put_caption(image=image_with_prediction, **online_prediction_result)
            # Add online prediction to a separate row
            results.append(
                [
                    image_with_prediction,
                ]
            )
        image_grid = concat_prediction_results(results=results)
        image_with_header = self._add_header_to_comparison(image_grid, target_device=target_device)

        # Save image to file
        cv2.imwrite(saved_image_path, cv2.cvtColor(image_with_header, cv2.COLOR_RGB2BGR))
        return image_with_header
