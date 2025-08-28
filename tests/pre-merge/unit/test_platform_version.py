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

from geti_sdk.platform_versions import GETI_10_VERSION, GETI_11_VERSION, GetiVersion


class TestGetiVersion:
    def test_init(self) -> None:
        """Test the initialization of GetiVersion"""
        v1 = GetiVersion("1.0.0-release-20220129184214")
        v2 = GetiVersion("2.3.4")
        assert v1.version == Version("1.0.0")
        assert v2.version == Version("2.3.4")
        assert v1.build_tag == "release"
        assert not v2.build_tag
        assert v1.time_tag == "20220129184214"
        assert not v2.time_tag

    def test_comparison(self) -> None:
        """
        Test parsing the version from a version string, for different release versions
        of the Geti platform. Also test comparisons between versions
        """
        assert GETI_10_VERSION < GETI_11_VERSION
        assert GETI_11_VERSION >= GETI_10_VERSION
