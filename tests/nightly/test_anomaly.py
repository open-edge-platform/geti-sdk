from geti_sdk import Geti
from geti_sdk.demos import ensure_trained_anomaly_project
from geti_sdk.rest_clients import ProjectClient
from tests.helpers import ProjectService, force_delete_project
from tests.helpers.constants import PROJECT_PREFIX
from tests.nightly.test_nightly_project import TestNightlyProject


class TestAnomaly(TestNightlyProject):
    PROJECT_TYPE = "anomaly"
    __test__ = True

    def test_project_setup(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_geti_no_vcr: Geti,
        fxt_project_client_no_vcr: ProjectClient,
    ):
        """
        Test the `ensure_trained_anomaly_project` method
        """
        project_name = f"{PROJECT_PREFIX}_ensure_trained_anomaly_project"
        existing_project = fxt_project_client_no_vcr.get_project_by_name(project_name)
        if existing_project is not None:
            force_delete_project(
                project=existing_project,
                project_client=fxt_project_client_no_vcr,
            )
        assert project_name not in [project.name for project in fxt_project_client_no_vcr.get_all_projects()]

        project = ensure_trained_anomaly_project(geti=fxt_geti_no_vcr, project_name=project_name)
        fxt_project_service_no_vcr._project = project

        assert fxt_project_service_no_vcr.prediction_client.ready_to_predict

    def test_monitor_jobs(self, fxt_project_service_no_vcr: ProjectService):
        """
        For anomaly projects, the training is run in the project_setup
        phase. No need to monitor jobs.
        """
        pass

    def test_upload_and_predict_image(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_image_path: str,
        fxt_geti_no_vcr: Geti,
    ):
        super().test_upload_and_predict_image(fxt_project_service_no_vcr, fxt_image_path, fxt_geti_no_vcr)

    def test_deployment(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
        fxt_image_path: str,
        fxt_image_path_complex: str,
        fxt_artifact_directory: str,
    ):
        super().test_deployment(
            fxt_project_service_no_vcr,
            fxt_geti_no_vcr,
            fxt_temp_directory,
            fxt_image_path,
            fxt_image_path_complex,
            fxt_artifact_directory,
        )
