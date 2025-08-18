# Examples for the Intel® Geti™ SDK

## Getting started
The example scripts provided here show several common usecases for the Intel® Geti™ SDK. To run
the examples, simply:
1. Install the `geti-sdk` package into your python environment
2. Create a `.env` file containing the authentication information for your Intel® Geti™
   server, following the instructions in the [Authentication](#authentication)
   box.
3. In your terminal, navigate to the `examples` folder
4. Activate your python environment
5. Run the example script you'd like to run using `python <name_of_script.py>`

> ### Authentication
>
> The example scripts rely on a `.env` file to load the server details for the Intel® Geti™
> instance which they run against. To provide the details for your Intel® Geti™ instance,
> create a file named `.env` directly in the `examples` directory. Two types of
> authentication are supported: Either via a Personal Access Token (the recommended
> approach) or via user credentials.
>
> #### Personal Access Token
> To use the personal access token for authenticating on your server, the `.env` file
> should have the following contents:
> ```shell
> # GETi instance details
> HOST=
> TOKEN=
> ```
> Where you should of course fill the details appropriate for your instance. For details
> on how to acquire a Personal Access Token, please refer to the section
> [Connecting to the Geti platform](../README.md#connecting-to-the-geti-platform) in the
> main readme.
>
> #### Credentials
> To use your user credentials for authenticating on your server, the `.env` file
> should have the following contents:
> ```shell
> # GETi instance details
> HOST=
> USERNAME=
> PASSWORD=
> ```
> Where you should of course fill the details appropriate for your instance.
>
> In case both a TOKEN and USERNAME/PASSWORD variables are provided, the SDK
> will default to using the TOKEN since this method is considered more secure.
> #### SSL Certificate verification
> By default, the SDK verifies the SSL certificate of your server before establishing
> a connection over HTTPS. If the certificate can't be validated, this will results in
> an error and the SDK will not be able to connect to the server.
>
> However, this may not be appropriate or desirable in all cases, for instance if your
> Geti server does not have a certificate because you are running it in a private
> network environment. In that case, certificate validation can be disabled by adding
> the following variable to the `.env` file:
> ```shell
> VERIFY_CERT=0
> ```

## Uploading and getting predictions for media
The example scripts `upload_and_predict_from_numpy.py` and
`upload_and_predict_media_from_folder.py` show how to upload either a single media
item directly from memory, or upload an entire folder of media items and
get predictions for the media from the cluster.

## Predict a video on local environment
Once you download(deploy) a model from the server, you can get predictions on the local environment.
The example script `predict_video_locally.py` shows how to reconstruct a video with overlaid predictions without uploading the file to server.

This code sample shows how to get a deployment from the server.

> ```shell
> # Get the server configuration from .env file
> server_config = get_server_details_from_env()
>
> # Set up the Geti instance with the server configuration details
> geti = Geti(server_config=server_config)
>
> # Create deployment for the project, and prepare it for running inference
> deployment = geti.deploy_project(project_name=PROJECT_NAME)
>
> # Save deployment on local
> deployment.save(PATH_TO_DEPLOYMENT)
> ```
