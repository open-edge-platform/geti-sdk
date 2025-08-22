import cv2
import numpy as np

from geti_sdk import Geti
from geti_sdk.demos import EXAMPLE_IMAGE_PATH, ensure_trained_example_project
from geti_sdk.utils import (
    get_server_details_from_env,
    show_image_with_annotation_scene,
    show_video_frames_with_annotation_scenes,
)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an input array representing an image by a certain `angle` (in degrees)

    :param image: Numpy array holding the pixel data for the image to rotate
    :param angle: Angle to rotate the image by, in degrees
    :return: Rotated image
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    return cv2.warpAffine(src=image, M=rotation_matrix, dsize=image.shape[1::-1], flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    # Get the server configuration from .env file
    server_config = get_server_details_from_env()

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up the Geti instance with the server configuration details
    geti = Geti(server_config=server_config)

    # `PROJECT_NAME` is the name of the project to which the media should be uploaded,
    # and from which predictions can be requested. A project with this name should
    # exist on the cluster. If the project exists but doesn't have any trained models,
    # the media will be uploaded but no predictions will be generated.
    PROJECT_NAME = "COCO dog detection"

    # `PATH_TO_IMAGE` is the path to the image that should be uploaded
    PATH_TO_IMAGE = EXAMPLE_IMAGE_PATH

    # `DELETE_AFTER_PREDICTION` can be set to True to delete the media from the
    # project once all predictions are downloaded. This can be useful to save disk
    # space on the cluster, or to avoid cluttering a project with a lot of
    # unannotated media
    DELETE_AFTER_PREDICTION = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # First, we load the image as a numpy array using opencv
    bgr_image = cv2.imread(PATH_TO_IMAGE)
    numpy_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Rotate the image by 20 degrees
    rotated_image = rotate_image(image=numpy_image, angle=20)

    # Make sure that the project exists
    project = ensure_trained_example_project(geti=geti, project_name=PROJECT_NAME)

    print(
        "Uploading and predicting example image now... The prediction results will be "
        "overlayed on the image and saved to a file in your current working directory."
    )

    # We can upload and predict the resulting array directly:
    sc_image, image_prediction = geti.upload_and_predict_image(
        project=project,
        image=rotated_image,
        visualise_output=False,
        delete_after_prediction=DELETE_AFTER_PREDICTION,
    )
    filepath = "example_prediction.jpg"
    # Save the result to a file
    show_image_with_annotation_scene(image=sc_image, annotation_scene=image_prediction, filepath=filepath)
    print(f"Prediction for the example image was saved to file: '{filepath}'")

    # We can do the same with videos. For example, to investigate the effect image
    # rotations have on our model predictions, we can create a video with rotated
    # versions of the image as frames. That will allow us to quickly see how well
    # our model holds up under image rotations

    # Create list of rotated images, these will be the frames of the video
    print(
        "Creating a video of rotated images, and requesting predictions for all "
        "frames... Output predictions will be written to file upon completion."
    )
    rotation_video = []
    for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        rotation_video.append(rotate_image(image=numpy_image, angle=angle))

    print("Video generated, retrieving predictions...")
    # Create video, upload and predict from the list of frames
    sc_video, video_frames, frame_predictions = geti.upload_and_predict_video(
        project=project,
        video=rotation_video,
        frame_stride=1,
        visualise_output=False,
        delete_after_prediction=DELETE_AFTER_PREDICTION,
    )

    video_filepath = "example_video_prediction.avi"
    # Again, we save the result to a file
    show_video_frames_with_annotation_scenes(
        video_frames=video_frames,
        annotation_scenes=frame_predictions,
        filepath=video_filepath,
    )
    print(f"Predictions for the video were saved to file: '{video_filepath}'")
