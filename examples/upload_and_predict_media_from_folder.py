import os

from geti_sdk import Geti
from geti_sdk.demos import EXAMPLE_IMAGE_PATH, ensure_trained_example_project
from geti_sdk.utils import get_server_details_from_env

if __name__ == "__main__":
    # Get the server configuration from .env file
    server_config = get_server_details_from_env()

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up the Geti instance with the server configuration details
    geti = Geti(server_config=server_config)

    # `FOLDER_WITH_MEDIA` is the path to the directory with images and videos that
    # should be uploaded to the GETi cluster
    FOLDER_WITH_MEDIA = os.path.dirname(EXAMPLE_IMAGE_PATH)

    # `PROJECT_NAME` is the name of the project to which the media should be uploaded,
    # and from which predictions can be requested. A project with this name should
    # exist on the cluster. If the project exists but doesn't have any trained models,
    # the media will be uploaded but no predictions will be generated.
    PROJECT_NAME = "COCO dog detection"

    # `DELETE_AFTER_PREDICTION` can be set to True to delete the media from the
    # project once all predictions are downloaded. This can be useful to save disk
    # space on the cluster, or to avoid cluttering a project with a lot of
    # unannotated media
    DELETE_AFTER_PREDICTION = False

    # `OUTPUT_FOLDER` is the target folder to which the predictions will be saved
    OUTPUT_FOLDER = "media_folder_predictions"

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # Make sure that the specified project exists on the server
    project = ensure_trained_example_project(geti=geti, project_name=PROJECT_NAME)

    # Upload the media in the folder and generate predictions
    geti.upload_and_predict_media_folder(
        project=project,
        media_folder=FOLDER_WITH_MEDIA,
        delete_after_prediction=DELETE_AFTER_PREDICTION,
        output_folder=OUTPUT_FOLDER,
    )
