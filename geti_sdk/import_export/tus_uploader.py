import os
import time
from io import BufferedReader

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.http_session.geti_session import GetiSession


class TUSUploader:
    """
    Class to handle tus uploads.
    """

    DEFAULT_CHUNK_SIZE = 5 * 2**20  # 5MB

    def __init__(
        self,
        session: GetiSession,
        base_url: str,
        file_path: os.PathLike | None = None,
        file_stream: BufferedReader | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        retries: int = 0,
        retry_delay: int = 30,
    ):
        """
        Initialize TUSUploader instance.

        :param session: GetiSession instance.
        :param base_url: Base url for the tus upload.
        :param file_path: Path to the file to be uploaded.
        :param file_stream: File stream of the file to be uploaded.
        :param chunk_size: Size of the chunk to be uploaded at each cycle.
        :param retries: Number of retries to be made in case of upload failure.
        :param retry_delay: Delay between retries.
        """
        if file_path is None and file_stream is None:
            raise ValueError("Either 'file_path' or 'file_stream' cannot be None.")

        self.file_path = file_path
        self.file_stream = file_stream
        if self.file_stream is None:
            self.file_stream = self.get_file_stream()
        self.stop_at = self.get_file_size()
        self.session = session
        self.base_url = base_url
        self.tus_resumable_version = self._get_tus_resumable_version()
        self.offset = 0
        self.upload_url = None
        self.chunk_size = chunk_size
        self.retries = retries
        self.request = None
        self._retried = 0
        self.retry_delay = retry_delay

    def _get_tus_resumable_version(self):
        """
        Return tus resumable version.
        """
        response = self.session.get_rest_response(
            url=self.base_url,
            method="OPTIONS",
        )
        return response.headers["tus-resumable"]

    def get_offset(self):
        """
        Return offset from tus server.

        Make an http request to the tus server to retrieve the current offset.

        :return: Offset value.
        :raises: Exception if offset retrieval fails.
        """
        response = self.session.get_rest_response(
            url=self.upload_url,
            method="HEAD",
            request_headers={
                "tus-resumable": self.tus_resumable_version,
            },
        )
        offset = response.headers.get("upload-offset")
        if offset is None:
            raise Exception("Attempt to retrieve offset failed")
        return int(offset)

    def get_request_length(self):
        """
        Return length of next chunk upload.
        """
        remainder = self.stop_at - self.offset
        return self.chunk_size if remainder > self.chunk_size else remainder

    def get_file_stream(self) -> BufferedReader:
        """
        Return a file stream instance of the upload.

        :return: File stream instance.
        :raises: ValueError if file_path is invalid.
        """
        if self.file_stream:
            self.file_stream.seek(0)
            return self.file_stream
        if self.file_path is not None and os.path.isfile(self.file_path):
            return open(self.file_path, "rb")
        raise ValueError(f"invalid file {self.file_path}")

    def get_file_size(self):
        """
        Return size of the file.
        """
        stream = self.get_file_stream()
        stream.seek(0, os.SEEK_END)
        return stream.tell()

    def upload(self, stop_at: int | None = None):
        """
        Perform file upload.

        Performs continous upload of chunks of the file. The size uploaded at each cycle is
        the value of the attribute 'chunk_size'.

        :param stop_at: Offset value at which the upload should stop. If not specified this
            defaults to the file size.
        """
        self.stop_at = stop_at or self.get_file_size()

        if not self.upload_url:
            self.upload_url = self.create_upload_url()
            self.offset = 0

        self.file_stream = self.get_file_stream()
        with (
            logging_redirect_tqdm(tqdm_class=tqdm),
            tqdm(
                total=self.stop_at >> 20,
                desc="Uploading file",
                unit="MB",
            ) as tbar,
        ):
            while self.offset < self.stop_at:
                self.upload_chunk()
                tbar.update((self.offset >> 20) - tbar.n)

    def create_upload_url(self):
        """
        Return upload url.

        Makes request to tus server to create a new upload url for the required file upload.
        """
        response = self.session.get_rest_response(
            url=self.base_url,
            method="POST",
            request_headers={
                "tus-resumable": self.tus_resumable_version,
                "upload-length": str(self.get_file_size()),
            },
        )
        upload_url = response.headers.get("location")
        if upload_url is None:
            raise ValueError("Upload url not returned by server")
        return upload_url

    def get_file_id(self) -> str | None:
        """
        Return file id from upload url.

        :return: File id.
        """
        if self.upload_url is None or len(file_id := self.upload_url.split("/")[-1]) < 2:
            # We get the file_id from the upload url. If the url is not set or the file_id
            # is not valid (may be an empty string if the url is not valid), we return None.
            return None
        return file_id

    def upload_chunk(self):
        """
        Upload chunk of file.
        """
        self._retried = 0
        try:
            self.offset = self._patch()
        except Exception as err:
            self._retry(err)

    def _patch(self) -> int:
        """
        Perform actual request.

        :return: Offset value after the request.
        """
        chunk = self.file_stream.read(self.get_request_length())
        # self.add_checksum(chunk)
        response = self.session.get_rest_response(
            url=self.upload_url,
            method="PATCH",
            data=chunk,
            contenttype="offset+octet-stream",
            request_headers={
                "upload-offset": str(self.offset),
                "tus-resumable": self.tus_resumable_version,
            },
        )
        upload_offset = int(response.headers.get("upload-offset"))
        return int(upload_offset)

    def _retry(self, error):
        """
        Retry upload in case of failure.

        :param error: Error that caused the upload to fail.
        :raises: error if retries are exhausted.
        """
        if self.retries > self._retried:
            time.sleep(self.retry_delay)

            self._retried += 1
            try:
                self.offset = self.get_offset()
            except Exception as err:
                self._retry(err)
            else:
                self.upload()
        else:
            raise error
