# v2.13.0 Geti™ SDK

## What's Changed

### ✨ Features

* Enable support for Python 3.13 by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/664
* Rebranding of Geti name and support for multiworkspace by @jcchr in https://github.com/open-edge-platform/geti-sdk/pull/666
  * handling of multiworkspace environments has been simplified
* Relaxing control of compatibility with Geti, improving export of shapes components by @jcchr in https://github.com/open-edge-platform/geti-sdk/pull/699

### 🐞 Bugfixes
* Fixed issue with export DatasetFormat symbol from data_models.enums by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/696

### 🔒 Security
* Adding Semgrep in the CI by @AlexanderBarabanov in https://github.com/open-edge-platform/geti-sdk/pull/680
* Updated several dependencies
  * pydantic by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/695
  * certifi by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/698
  * pillow by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/691
  * cryptography by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/676

## New Contributors
* @jcchr made their first contribution in https://github.com/open-edge-platform/geti-sdk/pull/666

**Full Changelog**: https://github.com/open-edge-platform/geti-sdk/compare/v2.12.0...v2.13.0

# v2.12.0 Geti™ SDK

## What's Changed

### ✨ Features
* New clients to update the configuration by @maxxgx in https://github.com/open-edge-platform/geti-sdk/pull/643
  * `ProjectConfigurationClient` allows to reconfigure project-level settings, such as auto-training
  * `TrainingConfigurationClient` allows to reconfigure training parameters (hyperparameters, subset split, ...)
* Added parameter `include_models` to `GetiIE.export_project` by @JortBergfeld in https://github.com/open-edge-platform/geti-sdk/pull/648
  * This option allows users to choose what set of models to include in the archive (supported values: `all`, `none` or `latest_active`)
* Updated notebooks scenarios to cover the new features by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/654

### 🐞 Bugfixes
* Fixed issue where `GetiIE.import_project` would sometimes fail to retrieve the project right after importing it by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/644
* Fixed issue where `ModelClient.get_all_model_groups` would fail with visual prompting models by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/661
* Fixed missing attribute in algorithm summary by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/652

### 🔒 Security
* Updated security CI workflows by @AlexanderBarabanov in https://github.com/open-edge-platform/geti-sdk/pull/659
* Enabled Renovate in CI by @AlexanderBarabanov in https://github.com/open-edge-platform/geti-sdk/pull/636
* Updated several dependencies
  * urllib3 and numpy by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/658
  * pathvalidate by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/641
  * pydantic by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/650 
  * certifi by @oep-renovate[bot] in https://github.com/open-edge-platform/geti-sdk/pull/653
  * pillow by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/627

### 🦖 Deprecation
* Dropped support for Python 3.9 by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/656
  * Python 3.9 is due to reach its scheduled upstream EOL in October 2025
* Deprecated the old `ConfigurationClient`

### ➕ Extra
* Rewritten README from scratch by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/660
  * Updated instructions to install and use the SDK
  * Updated instructions to run the notebooks
  * Extended code samples to cover more use cases
* Significantly reduced the number of required runtime dependencies by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/654
  * This should also solve some installation problems on Windows/MacOS
* Updated `DeploymentClient` to use the new deployment endpoints by @AlbertvanHouten in https://github.com/open-edge-platform/geti-sdk/pull/621
* Removed legacy check on deployed model version by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/646
* Updated ruff configuration & fixed linting issues by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/657

## New Contributors
* @AlbertvanHouten made their first contribution in https://github.com/open-edge-platform/geti-sdk/pull/621

**Full Changelog**: https://github.com/open-edge-platform/geti-sdk/compare/v2.11.0...v2.12.0


# v2.11.0 Geti™ SDK (15-07-2025)

## What's Changed

### ✨ Features
* Added support for asyncronous preprocessing of media uploaded in Geti by @warrkan in https://github.com/open-edge-platform/geti-sdk/pull/612

### 🔒 Security
* Hardened GHA workflows by @AlexanderBarabanov in https://github.com/open-edge-platform/geti-sdk/pull/603

### 🦖 Deprecation
* Deprecated Datumaro dataset and annotation reader by @JortBergfeld in https://github.com/open-edge-platform/geti-sdk/pull/616
  * The classes now raise a deprecation warning, and will be removed in the following release

### ➕ Extra
* Adopted uv for dependency management by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/609
* Replaced most linters with ruff by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/607
* Replaced call to deprecated deployment endpoint by @AlbertvanHouten in https://github.com/open-edge-platform/geti-sdk/pull/621
* Removed legacy payload from optimization requests by @JortBergfeld in https://github.com/open-edge-platform/geti-sdk/pull/605
* Removed some unnecessary dependencies by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/610
  * More deps will be removed in the next releases

## New Contributors
* @JortBergfeld made their first contribution in https://github.com/open-edge-platform/geti-sdk/pull/605
* @warrkan made their first contribution in https://github.com/open-edge-platform/geti-sdk/pull/612

**Full Changelog**: https://github.com/open-edge-platform/geti-sdk/compare/v2.10.2...v2.11.0


# v2.10.2 Geti™ SDK (04-07-2025)
## What's Changed

Bugfixes:
* Keypoints would not be displayed correctly in demo code by @A-Artemis in https://github.com/open-edge-platform/geti-sdk/pull/617

**Full Changelog**: https://github.com/open-edge-platform/geti-sdk/compare/v2.10.1...v2.10.2


# v2.10.1 Geti™ SDK (30-06-2025)
## What's Changed

Bugfixes:
* Fixed problem that would make keypoint deployment crash after label renaming by @maxxgx in https://github.com/open-edge-platform/geti-sdk/pull/613

**Full Changelog**: https://github.com/open-edge-platform/geti-sdk/compare/v2.10.0...v2.10.1


# v2.10.0 Geti™ SDK (30-05-2025)
## What's Changed

Features:
* Add new method to update the label colors and hotkeys by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/598
* Added experimental support for keypoint detection by @A-Artemis in https://github.com/open-edge-platform/geti-sdk/pull/588

Bugfixes:
* Solved proxy issue in demo data downloader by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/575
* Solved issue with incompatible pytest-recording version by @A-Artemis in https://github.com/open-edge-platform/geti-sdk/pull/590
* Fixed unrecognized JobType for import to existing dataset by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/594
* Fixed unrecognized 'lifecycle_stage' key in model group response by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/593
* Solved issue with missing prerelease version tag by @leoll2 and @tiepoint-bjornrun in https://github.com/open-edge-platform/geti-sdk/pull/596

Security:
* Enabled zizmor linter to check GHA workflows by @AlexanderBarabanov in https://github.com/open-edge-platform/geti-sdk/pull/576
* Enabled certificate validation for mvtec dataset download by @AlexanderBarabanov in https://github.com/open-edge-platform/geti-sdk/pull/589

Extra:
* Added PR template by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/579
* Added template for reporting bugs by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/582
* Added template for feature requests by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/583
* Added template for questions by @leoll2 in https://github.com/open-edge-platform/geti-sdk/pull/584

## New Contributors
* @A-Artemis made their first contribution in https://github.com/open-edge-platform/geti-sdk/pull/590
* @tiepoint-bjornrun made their first contribution in https://github.com/open-edge-platform/geti-sdk/pull/596

**Full Changelog**: https://github.com/open-edge-platform/geti-sdk/compare/v2.8.0...v2.10.0


# v2.9.0 Geti™ SDK (skipped)


# v2.8.0 Geti™ SDK (17-03-2025)
## What's Changed

* Bugfix: inference would sometimes fail if labels have number-like names by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/567
* Bugfix: inference would sometimes fail due to incorrect handling of empty label by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/566
* Removed legacy Dockerfile by @gdlg in https://github.com/openvinotoolkit/geti-sdk/pull/568
* Security fixes by @AlexanderBarabanov and @maxxgx

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.7.1...v2.8.0


# v2.7.1 Geti™ SDK (26-02-2025)
## What's Changed

* Bugfix: automatic workspace selection would sometimes not work by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/565

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.7.0...v.2.7.1


# v2.7.0 Geti™ SDK (17-02-2025)
## What's Changed

* Added support for Python 3.12 by @gdlg in https://github.com/openvinotoolkit/geti-sdk/pull/549
* Bugfix: models with 'tiling' enabled would sometimes return no predictions by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/547
* Bugfix: some notebooks would display predictions with inverted color channels by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/556
* Deprecated inference through OpenVINO model server (OVMS) by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/557
* Deprecated hyper-parameter optimization (HPO) by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/542

## New Contributors
* @AlexanderBarabanov made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/541
* @gdlg made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/549

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.6.2...v.2.7.0


# v2.6.2 Geti™ SDK (08-01-2025)
## What's Changed
* Bugfix: inference not working for classification projects with label containing spaces in their name by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/539

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.6.1...v2.6.2


# v2.6.1 Geti™ SDK (02-01-2025)
## What's Changed
* Bugfix: empty label sometimes not recognized during inference by @maxxgx in https://github.com/openvinotoolkit/geti-sdk/pull/534

## New Contributors
* @maxxgx made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/534

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.6.0...v2.6.1


# v2.6.0 Geti™ SDK (26-12-2024)
## What's Changed
* Updated model representation to that of Geti V2.6 by @Daankrol in https://github.com/openvinotoolkit/geti-sdk/pull/523
* Mark model response fields as deprecated in 2.6 instead of removed by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/528
* Fix VCR config for unset proxy by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/527
* Tolerate extra keys when deserializing HTTP response by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/529
* CVS-154056 Update training/optimization endpoint paths by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/530

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.5.0...v2.6.0


# v2.5.0 Geti™ SDK (22-10-2024)
## What's Changed
* Introduce `delete_dataset` method  by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/489
* Update the download_all method for ImageClient to download specific dataset by @rajeshgangireddy in https://github.com/openvinotoolkit/geti-sdk/pull/484
* Update media info schema to include extension by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/494
* Project access refactoring by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/481
* Images upload from list optimization by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/495
* Add readthedocs configuration file by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/497
* Fix bug with label name mapping for hierarchical classification models by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/498
* Introduce new `Anomaly` task by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/487
* Implement `get_training_dataset` method in `DatasetClient` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/492
* Fix anomaly task title by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/499
* Anomaly tasks backwards compatibility by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/502
* Fix anomaly demo test by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/501
* Introduce project removal delay for short tests by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/500
* Update datumaro requirement from ==1.8.* to ==1.9.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/496
* Update pytest requirement from ==8.2.* to ==8.3.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/482
* Fix organization id fetching by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/504
* Update ModelGroup data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/506
* Update Model data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/508
* Update pillow requirement from ==10.4.* to ==11.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/507
* Fix label assignment for deployed models by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/505

## New Contributors
* @rajeshgangireddy made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/484

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.3.0...v2.5.0


# v2.3.0 Geti™ SDK (03-09-2024)
## What's Changed
* Add method to purge models by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/468
* Fix visualization in 008 example notebook by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/472
* Update the Model schema to include the `total_disk_size` field  by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/471
* Add support for Credit System API by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/473
* Implement Platform version check by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/474
* Allow Credit System-less servers to authenticate with a token by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/476
* Empty label visualization fix by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/480


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.2.0...v2.3.0


# v2.2.0 Geti™ SDK (18-07-2024)
## What's Changed
* Add `description` attribute to the job class by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/448
* Project/Dataset export import API alignment by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/446
* Create geti_sdk.import_export docs page by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/451
* Introduce data_model `purge_info` field by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/453
* Disable certificate validation for mvtec dataset download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/452
* Use proper project name in notebook 008 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/454
* Update pillow requirement from ==10.3.* to ==10.4.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/449
* Create workflow to build wheels for different OS/CPU architecture combinations by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/456
* Fix headers in `HttpRequest` PostInferenceAction by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/457
* Correctly set headers in `HttpRequest` PostInferenceAction by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/458
* Expose Visualizer class by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/426
* Job and Model schema update by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/459
* Use `json` instead of `data` parameter in HttpRequestAction by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/460
* Remove workflows that require a self-hosted runner by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/461
* Fix 005_modify_image.ipynb image usage by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/463
* Fix permissions for wheel build workflow and remove record_cassettes workflow by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/462
* Fix `delete_project` method by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/466
* Add a Custom Dataset upload example by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/464
* Add Job Cost info to job data scheme by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/467


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/releases/v2.1.0...1234

# v2.1.0 Geti™ SDK (19-06-2024)
## What's Changed
* Saliency map visualization by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/424
* Add a model management notebook by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/419
* Add support for asynchronous inference in deployments by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/421
* Bump imageio-ffmpeg from 0.4.9 to 0.5.1 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/430
* Saliency map fix by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/427
* Fix rotated detectionsaliency map processing by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/431
* Handle projects with duplicate names in the `ProjectClient` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/425
* Update opencv-python requirement from ==4.9.* to ==4.10.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/428
* Update nightly test conda env directory by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/432
* Fix `is_geti` version check by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/434
* Fix build and publish workflow for internal pypi by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/433
* Add check to make sure project is trained in notebook 012 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/439
* Update flake8 requirement from ==7.0.* to ==7.1.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/440
* Fix removal of empty strings in json data for POSTing by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/442
* Include `model_group_id` in `Model` serialization and deserialization by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/443
* Include `model_group_id` upon model deployment in DeploymentClient by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/444
* Small fixes for testing client and test result by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/438
* Run pre-merge tests on PR to release branch by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/436
* Fix model_group_id assignment in DeployedModel deserialization by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/445
* Fix bug in `AsyncVideoProcessor` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/441


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.0.1...v2.1.0

# v2.0.1 Geti™ SDK (29-05-2024)
## What's Changed
* Add retry mechanism to better handle `ConnectionError` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/420
* Undo changes to video tempfile handling, register atexit handler for tempfile deletion by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/423


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v2.0.0...v2.0.1

# v2.0.0 Geti™ SDK (16-05-2024)
## New features
This release introduces a new feature related to model deployment: post-inference hooks! A post-inference hook can be added to any `Deployment`, and will be executed after every inference request (i.e. every call to `deployment.infer()`). The hooks allow you to define specific actions to take under certain conditions. For example, a hook could implement the following behaviour:
**If** the confidence level of one of the predictions for the image is less than 20%, **then** upload the image to the Geti™ project in which the model was trained.

This could be useful for improving your model with a next training round, because including such 'low confidence images' in the training dataset might help to improve model accuracy.
Additional examples of post-inference hooks, and instructions for configuring them, can be found in the newly added [notebook 012](https://github.com/openvinotoolkit/geti-sdk/blob/main/notebooks/012_post_inference_hooks.ipynb) in this repository.

## Breaking changes
This major release of the Geti™ SDK breaks backwards compatibility with Geti™ servers of version v1.14 and below. Please make sure that your Geti™ server is updated to the latest version of the Geti™ platform, to prevent compatibility issues.

## What's Changed
* Update `Video` data model with annotation statistics by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/391
* Record Legacy and Develop cassette in separate steps instead of pipelines by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/387
* Remove dependency on OTX by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/393
* Add `model_storage_id` to models when fetching model group by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/399
* Add Python 3.11 support; Deprecate Python 3.8 by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/398
* Add `last_annotator_id` field to media data model by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/403
* Documentation Update 2.0 by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/402
* End support for Platforms versions lower than 1.15 by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/397
* Bump imageio-ffmpeg from 0.4.8 to 0.4.9 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/404
* Update python version in github workflows by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/405
* Update requirement for tqdm to `>=4.66.3` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/407
* Add `PostInferenceHook` feature initial implementation and notebook by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/355
* [StepSecurity] Apply security best practices by @step-security-bot in https://github.com/openvinotoolkit/geti-sdk/pull/408
* Define permissions on job level for cassette record workflow by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/409
* Update pytest-recording requirement from ==0.12.* to ==0.13.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/406
* Update pytest requirement from ==7.4.* to ==8.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/401
* Update cryptography requirement to `>=42.0.7` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/410
* Update requirement for joblib to `>=1.2.0` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/411
* Update certifi requirement to `>=2024.2.2` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/412
* Update ipython requirement from ==8.12.* to ==8.18.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/400
* Update vcrpy requirement from ==4.4.* to ==6.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/318
* Remove dead code in training client by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/413
* Update requests requirement from ==2.31.* to ==2.32.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/415
* Update pytest-html requirement from ==3.2.* to ==4.1.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/418
* Update pytest-cov requirement from ==4.1.* to ==5.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/416
* Add initial support for Tiling inference [synchronous mode only] by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/414

## New Contributors
* @step-security-bot made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/408

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.16.1...v2.0.0


# v1.16.1 Geti™ SDK (22-04-2024)
## What's Changed
* Add `default_workspace` to possible default workspace names by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/394


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.16.0...v1.16.1

# v1.16.0 Geti™ SDK (26-03-2024)
## What's Changed
* Decouple visualizer for OTX by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/356
* Update data models to account for REST API changes in Geti v1.16 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/357
* Fix organization retrieval URL with PAT by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/361
* TEST: add VCR recording workflow by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/360
* Fix job id retrieval in optimize_model method by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/364
* Update job datamodel by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/362


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.15.0...v1.16.0

# v1.15.0 Geti™ SDK (12-03-2024)
This release makes the SDK compatible for Geti™ v1.15. The majority of the changes is focused on that, but below is a list of other changes that are worth mentioning.

## Release Highlights
- Add compatibility for Geti™ v1.15. Note that we maintain backwards compatibility for models created in Geti™ v1.8: This means that the SDK can run inference on deployments from both both the latest Geti™ on-prem release and Geti™ SaaS.
- Add benchmarking functionality for deployments through the `Benchmarker` class, as well as a new notebook to demonstrate this feature.
- Improve the job monitoring feature, job progression is now displayed via progress bars.
- Image and annotation upload and download now uses multithreading, greatly speeding up the process.
- A new notebook demonstrating an end-to-end workflow, from model creation to deployment, was added to `notebooks/use_cases`.
- Multiple dependency updates, minor bug fixes and documentation improvements.

## What's Changed
* Add organization ID to session when appropriate by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/294
* Change job monitor functions to use progress bars by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/295
* Decrease verbosity for annotation download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/296
* Improve model selection mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/297
* Add `Benchmarker` class and demo notebook for inference throughput benchmarking of Geti models by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/300
* Supported algorithms workaround by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/303
* Workaround for getting supported algorithms by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/305
* Update README.md by @operepel in https://github.com/openvinotoolkit/geti-sdk/pull/307
* Fix organization ID when uploading annotations by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/306
* Mention git lfs in test docs by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/308
* Fix inference endpoints and job data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/309
* Improve handling of supported algorithms by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/310
* Fix TaskAnnotationState and Optimization JobType by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/312
* Test benchmarker by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/311
* Pin otx to v1.4.3 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/313
* Remove snyk scan from security workflow by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/314
* Minor bug fixes related to deployment by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/315
* Inference results visual comparison by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/316
* Add `model_activated` field to ModelMetadata by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/320
* Update `Pillow` dependency to `Pillow==10.2.*` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/322
* Add shields.io badges to readme by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/321
* Relax time requirement for demo projects nightly tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/325
* Multithread image upload and download by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/323
* Handle the new authorization mechanism and organization IDs by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/326
* Improve robustness of job monitoring by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/327
* Introduce method to switch active model by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/328
* Fix issue with undefined label color for prediction by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/330
* Enable pre-merge tests on Windows and MacOS by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/331
* Update compatibility table in README by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/332
* Add compatibility for anomaly model deployment for Geti v1.13 and up by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/333
* Correctly handle errors in new dex authentication mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/336
* Handle the new personal access tokens for Geti v1.15 and up by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/335
* Fix proxies issue in GetiSession by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/334
* Modified 'Smart Cities' notebook by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/304
* Fix annotation kind comparison in plot_helpers by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/337
* Update flake8 requirement from ==6.0.* to ==7.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/302
* Handle org ID with personal access tokens for Geti SaaS by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/338
* Bump orjson from 3.9.2 to 3.9.15 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/344
* Update pytest-env requirement from ==1.0.* to ==1.1.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/341
* Update datumaro requirement from ==1.4.* to ==1.5.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/347
* geti_session - remove unused proxy kwargs by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/346
* CVS-117852 Fix bandit issues by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/348
* Make `datasets` optional in Project data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/339
* Fix maskrcnn postprocessing for geti v1.15 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/350
* Restrict default top-level permissions for all workflows by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/349
* Add openssf scorecard code scanning by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/351
* OTX decoupling part 1: inference postprocessing by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/352
* OTX decoupling: deployment by @igor-davidyuk in https://github.com/openvinotoolkit/geti-sdk/pull/345

## New Contributors
* @operepel made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/307
* @igor-davidyuk made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/308

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v.1.8.2...v1.15.0

# v1.8.2 Geti™ SDK (22-02-2024)
## What's Changed
* Update `opencv-python` requirement to `4.9.*`
* Update `Pillow` requirement to `10.2.*`
* Update `otx` requirement to `4.4.4`
* Add backwards compatibility for anomaly model deployment


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v.1.8.1...v.1.8.2

# v1.8.1 Geti™ SDK (20-11-2023)
## What's Changed
* Update pytest requirement from ==7.3.* to ==7.4.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/261
* Update pytest-env requirement from ==0.8.* to ==1.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/275
* Update various dependencies by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/288


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v.1.8.0...v.1.8.1

# v1.8.0 Geti™ SDK (16-10-2023)
## What's Changed
* Predict video on local by @jihyeonyi in https://github.com/openvinotoolkit/geti-sdk/pull/243
* Update job datamodel for new job scheduler by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/251
* Add `model` key to TestMetaData by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/252
* Improve error handling for version parsing by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/253
* Update SECURITY.md by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/256
* Add `nosec` for safe subprocess use in `predict_video.py` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/257
* Update openvino to 2023.0 and OTX to v1.4 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/255
* Update opencv-python requirement from ==4.5.* to ==4.8.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/249
* Bump orjson from 3.8.8 to 3.9.2 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/250
* Enable using pre-production dependencies in test builds by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/258
* Update the list of third party programs by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/259
* Require `cryptography>=41.0.2` for security reasons by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/260
* Fix for responding to Project Key values in the REST API by @harimkang in https://github.com/openvinotoolkit/geti-sdk/pull/264
* Disable parallel execution on pre-merge tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/270
* Fix model_api import and model creation for deployments by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/268
* CVS-116946 Make platform version parsing robust by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/269
* Update Performance attribute interfaces by @harimkang in https://github.com/openvinotoolkit/geti-sdk/pull/271
* CVS-118292 Update ATSS algo name by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/272
* Fix deployment postprocessing by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/273
* Fix OVMS configuration generation for Geti v1.8 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/276
* Update data model for `Algorithm` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/278
* Update configurable parameter data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/279
* correct H1 level heading by @adamczap in https://github.com/openvinotoolkit/geti-sdk/pull/280
* Update `TestResult` data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/281
* Fix a potential infinite loop in the label helpers by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/285


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.8...v.1.8.0

# v1.5.8 Geti™ SDK (19-06-2023)
## What's Changed
* Update vcrpy requirement from ==4.2.* to ==4.3.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/227
* Specify correct project name in notebook 009 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/229
* Fix name of project to download in notebook 009 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/233
* Update pytest-cov requirement from ==4.0.* to ==4.1.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/231
* upgrate isort to 5.12.0 to avoid runtime error in pre-commit in python 3.8 by @jihyeonyi in https://github.com/openvinotoolkit/geti-sdk/pull/235
* Update all pre-commit hooks to latest versions by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/237
* Add `step_size` field to configurable floats by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/241
* Handle re-authentication during media upload robustly by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/244

## New Contributors
* @jihyeonyi made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/235

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.7...v1.5.8

# v1.5.7 Geti™ SDK (30-05-2023)
## What's Changed
* Allow more efficient image uploading for datumaro annotation readers by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/224
* Fix deployment for `otx v1.2.2` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/225


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.6...v1.5.7

# v1.5.6 Geti™ SDK (23-05-2023)
## What's Changed
* Add `group` key to hierarchical label definition in notebook 001 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/220
* Add `TestingClient` to perform model tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/221
* Wait for a project to become ready after it is created by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/223
* Update requests requirement from ==2.28.* to ==2.31.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/222


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.5...v1.5.6

# v1.5.5 Geti™ SDK (15-05-2023)
## What's Changed
* Add param to disable certificate validation for data download helpers by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/218


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.4...v1.5.5

# v1.5.4 Geti™ SDK (11-05-2023)
## What's Changed
* Add active learning client for retrieving the active set by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/215
* Allow passing label dictionaries to `Geti.create_single_task_project_from_dataset` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/216
* Add string representation for `GetiVersion` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/217


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.3...v1.5.4

# v1.5.3 Geti™ SDK (10-05-2023)
## What's Changed
* Add `ONNX` as optimization type by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/212
* Remove trailing slash from the base url in the media rest client by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/214
* Update pytest requirement from ==7.2.* to ==7.3.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/211
* Fix nightly tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/213


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.2...v1.5.3

# v1.5.2 Geti™ SDK (08-05-2023)
## What's Changed
* Add score NoneType check in summary function of ProjectStatus by @harimkang in https://github.com/openvinotoolkit/geti-sdk/pull/208
* Update `OptimizedModel` data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/209
* Update datumaro requirement from ==1.1.* to ==1.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/206

## New Contributors
* @harimkang made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/208

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.1...v1.5.2

# v1.5.1 Geti™ SDK (26-04-2023)
## What's Changed
* Update hashing algorithm to `sha3_512` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/204
* Bump otx from 1.1.2 to 1.2.0 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/203
* Update ipython requirement from ==8.11.* to ==8.12.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/202


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.5.0...v1.5.1

# v1.5.0 Geti™ SDK (24-04-2023)
## What's Changed
* Pin orjson version to 3.8.8 to avoid installation error by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/178
* Fix nightly tests for classification project by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/179
* Add workaround to set output blob name if not set by model adapter by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/180
* Add workflow to publish to TestPyPI by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/185
* Filter on polygons when setting Datum segmentation dataset by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/184
* Add `explain` method for deployment by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/183
* Remove duplicate annotations for datumaro dataset items by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/186
* Remove specific geti version tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/187
* Add python version compatibility table to README by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/189
* Add test for python 3.10 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/188
* Use absolute URLs for links in readme by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/190
* Fix explain for segmentation and anomaly models by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/191
* Replace the deprecated `DatasetItem.image()` in DatumAnnotationReader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/192
* Update datumaro requirement from ==1.0.* to ==1.1.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/175
* Update pillow requirement from ==9.4.* to ==9.5.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/182
* Pagination for project fetching by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/197
* Use pagination when fetching all the projects by @leoll2 in https://github.com/openvinotoolkit/geti-sdk/pull/193
* Fix dataset filtering issue for Datumaro annotation reader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/194
* Update simplejson requirement from ==3.18.* to ==3.19.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/196
* Bump otx from 1.1.0 to 1.1.2 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/195
* Update omegaconf requirement from ==2.1.* to ==2.3.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/120
* Add finalizer to remove demo project after nightly test by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/198
* Add support for additional datasets in project by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/199

## New Contributors
* @leoll2 made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/193

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.4.1...v1.5.0

# [Pre-release] v1.4.1 Geti™ SDK (28-03-2023)
## What's Changed
* Update otx requirement to `otx==1.1.0` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/176
* Make model wrapper module namespace unique by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/177


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.4.0...v1.4.1

# [Pre-release] v1.4.0 Geti™ SDK (27-03-2023)
## What's Changed
* Migrate from `ote_sdk` to `otx.api` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/166

## Caution: Backwards incompatibility for project `Deployments`
This release breaks backwards compatibility with `deployments` created by earlier versions of the Geti™ platform. Please only update to this version of the Geti SDK if you are sure that your Geti™ server is also on version 1.4 or later.

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.2.4...v1.4.0

# v1.2.4 Geti™ SDK (27-03-2023)
## What's Changed
* Update ipython requirement from ==8.10.* to ==8.11.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/172
* Fix `upload_and_predict_from_numpy.py` example by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/173
* Improve error handling for OVMS deployment by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/174


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.2.3...v1.2.4

# v1.2.3 Geti™ SDK (13-03-2023)
## What's Changed
* Fix saving images and annotations with non-ascii characters in their filename by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/160
* Update tqdm requirement from ==4.64.* to ==4.65.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/161
* Cvs-96625 Update image display method in `show_image_with_annotation_scene` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/164
* Enable uploading data from nested subsets in DatumAnnotationReader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/163
* Update datumaro requirement from ==0.4.* to ==1.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/155
* Update OptimizedModel and TaskStatus data models by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/165
* Formatting images and cell - Notebook 102 suggestions by @paularamo in https://github.com/openvinotoolkit/geti-sdk/pull/167
* Add notebook `102_from_zero_to_hero` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/159
* Demo Zero to Hero in 9 steps by @paularamo in https://github.com/openvinotoolkit/geti-sdk/pull/154
* Fix deployment saving mechanism and handle errors by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/170
* Add ovmsclient to base requirements by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/169
* Issue 168: Fix tempdir cleanup in `DeployedModel` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/171

## New Contributors
* @paularamo made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/167

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.2.2...v1.2.3

# v1.2.2 Geti™ SDK (28-02-2023)
## What's Changed
* Enable OVMS deployment by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/148
* Minor fixes for notebook 010 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/149
* Update ipython requirement from ==8.9.* to ==8.10.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/151
* Add performance hint to the ovms config by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/152
* Fix bug in deployment resource clean up method by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/153
* Update python-dotenv requirement from ==0.21.* to ==1.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/156
* Add a short sleep in `Geti.upload_project_data` after media upload by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/157
* Add OVMS deployment resources to manifest by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/158


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.2.1...v1.2.2

# v1.2.1 Geti™ SDK (02-02-2023)
## What's Changed
* Fix issue with deployment for anomaly classification models by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/144
* Require `mistune>=2.0.3` for notebooks by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/147
* Update ipython requirement from ==8.8.* to ==8.9.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/145
* Fix issue with temporary resource clean up for `Deployment`s by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/146


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.2.0...v1.2.1

# v1.2 Geti™ SDK (24-01-2023)
## What's Changed
* Add `size` field to MediaInformation data model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/133
* Update available Geti versions by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/134
* Update pillow requirement from ==9.3.* to ==9.4.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/131
* Fix a mismatch in the data model for the Optimization Job by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/136
* Add DeploymentClient to streamline deployment mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/135
* Update prediction mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/137
* Update ipython requirement from ==8.7.* to ==8.8.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/138
* Bump openvino from 2022.2.0 to 2022.3.0 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/130
* Update various dependencies by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/140
* Minor refactor in `Geti` class and fix in `predict_image` method by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/139
* Make `maps` key optional when converting Predictions from dictionary input by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/141
* Fix documentation workflow to deploy from build artifact by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/143
* Make workaround for issue with detection prediction conversion by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/142


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.1.1...v1.2.0

# v1.1.1 Geti™ SDK (20-12-2022)
## What's Changed
* Fix issue with model to dictionary conversion by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/128
* Only submit train request once all running jobs for task have finished by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/129


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.1.0...v1.1.1

# v1.1.0 Geti™ SDK (15-12-2022)
## What's Changed
* Minor fix in README.md by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/118
* Fix and improve geti version comparison mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/117
* Ignore some false positive bandit detections by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/119
* Update datumaro requirement from ==0.3.* to ==0.4.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/121
* Add ClamAV workflow by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/123
* Update `jupyterlab` requirement for notebooks to >=3.5.1 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/124
* Increase timeout for training job polling upon calling `train_task` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/125


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.0.4...v1.1.0

# v1.0.4 Geti™ SDK (08-12-2022)
## What's Changed
* Update ipython requirement from ==8.6.* to ==8.7.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/112
* Properly check for empty annotation before uploading by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/111
* Update numpy version to 1.22.* in requirements by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/104
* Minor update to README.md by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/114
* Update training client to handle new /train endpoint response by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/113
* Update `StatusSummary` datamodel by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/115


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.0.3...v1.0.4

# v1.0.3 Geti™ SDK (25-11-2022)
## What's Changed
* Add `ScoreMetadata` to represent the new `scores` field by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/106
* Add model and prediction client integration tests + update cassettes by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/107
* Update simplejson requirement from ==3.17.* to ==3.18.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/108
* Fix opencv window closure bug by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/109


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.0.2...v1.0.3

# v1.0.2 Geti™ SDK (17-11-2022)
## What's Changed
* Update ote-sdk requirement to v0.3.1 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/98
* Add integration tests for `project_client`, fix `project_client.add_labels` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/99
* Update data model for TaskMetadata, improve robustness of active model fetching by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/100
* Use OTE SDK visualizer, add plot helper unit tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/101
* Add HTTPS_PROXY as variable to the credentials helper by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/102
* Add tests for video up- and download in Geti integration tests by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/103


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.0.1...v1.0.2

# v1.0.1 Geti™ SDK (11-11-2022)
## What's Changed
* Add path validation to project download target path by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/87
* Update tqdm requirement from ==4.62.* to ==4.64.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/84
* Update python-dotenv requirement from ==0.20.* to ==0.21.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/86
* Add security note to README for project download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/88
* Update numpy requirement to 1.21.* by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/89
* Reduce permissions upon directory creation by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/90
* Update README to correctly reference Geti brand everywhere by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/92
* Improve check for video processing in `Geti.upload_project_data()` to avoid potential infinite loop by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/93
* Add unit tests to pre-merge test suite by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/91
* Update ProjectStatus and TaskStatus to include new field `n_new_annotations` by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/94
* Add progress bars for up/download of projects, media, annotations and predictions by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/95
* Add ipywidgets to notebook requirements by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/97


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v1.0.0...v1.0.1

# v1.0.0 Geti™ SDK (04-11-2022)
## What's Changed
* Add a re-authentication mechanism when using token authentication by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/72
* Update pytest requirement from ==7.1.* to ==7.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/73
* Update pillow requirement from ==9.2.* to ==9.3.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/74
* Update pytest-html requirement from ==3.1.* to ==3.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/75
* Catch value error when invalid datetime string is converted by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/76
* Update nightly test workflow to include tests against Geti `develop` branch by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/78
* Enable SSL certificate validation by default by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/77
* Remove disallowed fields from project before POSTing by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/79
* Ignore false positive bandit detections by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/80
* Update numpy, ipython, jupyterlab versions by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/81
* Fix and unify folder naming for project download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/82


## Breaking Changes
* The `Geti` class now has SSL certificate validation enabled by default. This will result in an error when connecting to a server with a certificate that can't be validated. Note that it is still possible to disable certificate validation by passing `verify_certificate = False` when initializing the `Geti` instance. Please note that disabling certificate validation is unsafe and should only be considered in a secure network environment.


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.4...v1.0.0rc1

# v0.2.4 Geti™ SDK (25-10-2022)
## What's Changed
* Auto detect normalized annotation files for GetiAnnotationReader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/63
* Fix version detection mechanism and add tests for GetiVersion by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/64
* Minor changes for backward compatibility with SCv1.1 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/67
* Enable proxies in ONLINE test mode by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/66
* Fix proxy config in tests for online mode by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/68
* Updated Attrs Classes and Fields by @HiteshManglani123 in https://github.com/openvinotoolkit/geti-sdk/pull/65
* Validate media filename upon download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/69
* Update pytest-env requirement from ==0.6.* to ==0.8.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/70
* Fix anomaly classification deployment by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/71

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.3...v0.2.4

# v0.2.3 Geti™ SDK (06-10-2022)
## What's Changed
* Remove VCR from nightly test for demos by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/54
* Improve nightly tests for `demos` module by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/55
* Update sc_annotation_reader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/56
* Add version to optimized model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/57
* Update SDK platform version parsing mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/58
* Add nightly tests against Geti-MVP by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/59
* Handle failed training jobs in notebook 007 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/60
* Update example script to store prediction results to file by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/61

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.2...v0.2.3

# v0.2.2 Geti™ SDK (04-10-2022)
## What's Changed
* Add coverage report to pre-merge and nightly test artifacts by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/47
* Correctly set permissions on extracted files for anomaly dataset by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/51
* Update pytest-cov requirement from ==3.0.* to ==4.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/50
* Update pillow requirement from ==9.1.* to ==9.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/48
* Workflow update: Run nightly and integration tests in one step and get coverage by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/53

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.1...v0.2.2

# v0.2.1 Geti™ SDK (30-09-2022)
## What's Changed
* Replace SC references in docstrings by Geti by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/33
* Change package name from `geti_sdk` to `geti-sdk`. Import names are unchanged by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/34
* Update vcrpy requirement from ==4.1.* to ==4.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/35
* Bump datumaro from 0.3 to 0.3.1 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/37
* Bump openvino from 2022.1.0 to 2022.2.0 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/38
* Update requests requirement from ==2.26.* to ==2.28.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/39
* Handle exceptions in data deserialization by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/40
* Fix image path in notebook 008 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/41
* Use personal access token instead of credential authentication by default by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/42
* Add image showing the personal access token menu to README by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/43
* Add nightly tests for `demos` module by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/44
* Add screenshot of jupyter lab landing page to README by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/45

## New Contributors
* @dependabot made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/35

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.0...v0.2.1

# v0.2.0 Geti™ SDK (27-09-2022)

This is the first official release of the Geti™ Software Development Kit (SDK).

The purpose of this SDK is twofold:

1. Provide an easy-to-use interface to the [Geti™ platform](www.geti.intel.com), to manipulate
Geti™ projects and other entities or automate tasks on the platform. All
of this from a Python script or Jupyter notebook.


2. Provide an API to deploy and run models trained on the Geti™ server on your local
machine. The SDK Deployment module provides a straightforward
route to create a deployment for your Geti™ project, save it to a local disk and run
it offline.

This SDK includes various example scripts and Jupyter notebooks which illustrate a
range of use cases for the SDK. Make sure to check them out if you're getting
started!
