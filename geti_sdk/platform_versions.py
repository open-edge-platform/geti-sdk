# Copyright (C) 2022 Intel Corporation
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

from packaging.version import Version
from semver import Version as SemVersion


class GetiVersion:
    """
    Version identifier of the Intel Geti platform
    """

    _GETI10_TIMETAG = "20220910154208"
    _SC11_TIMETAG = "20220624125113"
    _SCMVP_TIMETAG = "20220129184214"

    def __init__(self, version_string: str) -> None:
        """
        Initialize the version

        :param version_string: String representation of the version, as returned by
            the Geti platform
        """
        version_parts = version_string.split("-")

        sem_version = SemVersion.parse(version_string)

        time_tag = version_parts[-1] if len(version_parts) >= 3 else ""
        base_version = Version(f"{sem_version.major}.{sem_version.minor}.{sem_version.patch}")
        build_tag = sem_version.prerelease.replace(f"-{time_tag}", "") if sem_version.prerelease else ""

        self.version = base_version
        self.build_tag = build_tag
        self.time_tag = time_tag

    def __gt__(self, other):
        """
        Return True if this GetiVersion instance is a later version than `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance corresponds to a later version of the Intel
            Geti platform than `other`
        """
        if not isinstance(other, GetiVersion):
            raise TypeError(f"Unsupported comparison operation, {other} is not a GetiVersion.")
        if self.version != other.version:
            return self.version > other.version
        return self.time_tag > other.time_tag

    def __lt__(self, other):
        """
        Return True if this GetiVersion instance is an earlier version than `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance corresponds to an earlier version of the Intel
            Geti platform than `other`
        """
        return not self >= other

    def __ge__(self, other):
        """
        Return True if this GetiVersion instance is a later or equivalent version than
        `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance corresponds to a later or equivalent version of
            the Intel Geti platform than `other`
        """
        return (self > other) or (self == other)

    def __le__(self, other):
        """
        Return True if this GetiVersion instance is an earlier or equivalent version
        than `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance corresponds to an earlier or equivalent version
            of the Intel Geti platform than `other`
        """
        return (self < other) or (self == other)

    def __eq__(self, other):
        """
        Return True if this GetiVersion instance is equal to `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance is equal to the GetiVersion passed in `other`
        """
        if not isinstance(other, GetiVersion):
            raise TypeError(f"Unsupported comparison operation, {other} is not a GetiVersion.")
        return self.version == other.version

    def __str__(self) -> str:
        """
        Return the Intel Geti version as a string

        :return: String containing the version
        """
        return f"{self.version}-{self.build_tag}-{self.time_tag}"

    def __repr__(self) -> str:
        """
        Return the string representation of the Intel Geti version

        :return: String representing the version
        """
        return f"GetiVersion({self.version}-{self.build_tag}-{self.time_tag})"

    @property
    def is_sc_mvp(self) -> bool:
        """
        Return True if the version corresponds to a platform on the SC MVP version of
        the software.
        """
        return self.version == Version("1.0.0") and self._SCMVP_TIMETAG <= self.time_tag <= self._SC11_TIMETAG

    @property
    def is_sc_1_1(self) -> bool:
        """
        Return True if the version corresponds to a platform on the SC v1.1 version of
        the software.
        """
        return self.version == Version("1.1.0") and self._SC11_TIMETAG <= self.time_tag <= self._GETI10_TIMETAG

    @property
    def is_configuration_revamped(self) -> bool:
        """
        Return True if the version corresponds to a platform with the revamped configuration (>=2.12).
        """
        return self.version >= Version("2.12.0")


SC_MVP_VERSION = GetiVersion("1.0.0-release-20220129184214")
SC_11_VERSION = GetiVersion("1.1.0-release-20220624125113")
GETI_10_VERSION = GetiVersion("1.0.0-release-20221005164936")
GETI_11_VERSION = GetiVersion("1.1.0-release-20221125121144")
GETI_12_VERSION = GetiVersion("1.2.0-release-20230101120000")
GETI_15_VERSION = GetiVersion("1.5.0-release-20230504111017")
GETI_18_VERSION = GetiVersion("1.8.0-release-20231018022911")
GETI_114_VERSION = GetiVersion("1.14.0-release-20240131095302")
GETI_116_VERSION = GetiVersion("1.16.0-release-20240320101320")
GETI_20_VERSION = GetiVersion("2.0.0-release-20240320101320")
GETI_22_VERSION = GetiVersion("2.2.0-release-20240320101320")
GETI_25_VERSION = GetiVersion("2.5.0-release-20240320101320")
