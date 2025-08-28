This directory contains the test suite for the Geti™ SDK. The tests are grouped
into two categories:

1. **Pre-merge tests** These are executed for every Pull Request to the Geti™
   SDK `main` branch. The suite contains both integration and unit tests, the main focus
   being the integration tests. The test code can be found in
   the [pre-merge](pre-merge) folder.

2. **Nightly tests** These tests are executed for the `main` branch every night at
   midnight (Amsterdam time). The nightly tests are end-to-end style tests, covering
   everything from project creation to local inference through model deployment. The
   test code can be found in the [nightly](nightly) directory.

# Pre-merge test suite
## Integration tests
The integration tests for the SDK leverage the recording of HTTP requests and responses,
relying on the VCR.py package. All integration tests are defined in the
[pre-merge/integration](pre-merge/integration) directory. By default, the tests are run
in offline mode, meaning that no actual Geti™ server is needed and no real
http requests are being made during testing. All requests are intercepted, and a
previously recorded response is returned. The recorded interactions can be found in
[fixtures/cassettes](fixtures/cassettes).
> **_NOTE:_**  You may need to fetch and checkout VCR cassette data using [Git Large File Storage (LFS)](https://git-lfs.com/). Make sure that you have the **git-lfs** package installed and run `git lfs pull` from the root repo directory to download the HTTP requests records.

## Unit tests
The SDK unit tests are defined in the [pre-merge/unit](pre-merge/unit) directory. The
tests are not designed to provide full coverage by themselves, but should be run in
conjunction with the integration tests. At this moment the unit tests focus on testing
exception flows (as opposed to happy flow) for methods that are hard to test via
integration testing.

# Nightly test suite
The nightly tests are defined in the [nightly](nightly) directory. They can only be run in
`ONLINE` mode, meaning that a live Geti™ server is required to run against them. The
nightly tests need to be run using a `online.ini` file that contains the host name and
login details for the Geti™ server to run the tests against (see section
[Running the tests](#running-the-tests) below).

# Running the tests
First, install the requirements for the test suite using
`pip install -r requirements/requirements-dev.txt`. Then, run the tests using
`pytest ./tests/pre-merge`, or
(optionally) enable coverage using `pytest --cov=geti_sdk ./tests/pre-merge`.

## Test modes
By default, the integration tests are executed in `OFFLINE` mode. In addition, they
can be run in `ONLINE` or `RECORD` mode. The simplest way to change modes is to
define a custom pytest configuration file for each mode, as explained below:

### Online mode
If you want to run the test suite against a live Geti™ server (for example to make sure
that the SDK data models are still up to date with the Geti™ REST contracts), the tests
can be executed in `ONLINE` mode. To do so, define a file `online.ini` in the `tests`
directory. This file should have similar content to the existing `offline.ini`, but
with the Geti™ server hostname and login details set appropriately:

> ```shell
> [pytest]
> env =
>   TEST_MODE=ONLINE
>   GETI_USERNAME=your_username
>   GETI_PASSWORD=your_password
>   GETI_HOST=https://your_geti_instance.com
> ```

### Record mode
If you have added a new test that makes HTTP requests, all cassettes should be deleted
and re-recorded to maintain consistency across the recorded responses. This can be done
by running the tests in `RECORD` mode. The easiest way to do this is to create a file
`record.ini` with the same contents as the `online.ini` file above, but set
`TEST_MODE=RECORD` instead of `ONLINE`.

> **WARNING**: Running the test suite in `ONLINE` or `RECORD` mode will increase the
> time required for test execution considerably.

> **WARNING**: VCR playback in `OFFLINE` mode only supports username/password authentication. 
> Avoid using token-based authentication in `RECORD` mode.

## Running the tests in a non-default mode
Once you created the custom `online.ini` or `record.ini` configurations, you can run
the tests using `pytest -c online.ini ./pre-merge`. This will execute the tests in
online mode.

### Tests Geti™ SDK against the previous version of Geti™ server.
You can run the test suite against a legacy version of Geti™ server.
- In `ONLINE` mode  Simply assign the `GETI_HOST` variable with the legacy server’s address.
- In `OFFLINE` and `RECORD` modes one can set the `GETI_PLATFORM_VERSION` variable to `LEGACY`
within `offline.ini` or `record.ini` correspondingly, thus utilizing the pre-recorded cassettes from the prior server release for testing.

> **_NOTE:_**  Currently,  Geti™ 1.8 is considered as the legacy release.
