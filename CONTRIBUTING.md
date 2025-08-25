We appreciate any contribution to the Intel® Geti™ SDK, whether it's in the form of a
Pull Request, Feature Request or general comment/issue that you found. For feature
requests and issues, please feel free to create a GitHub Issue in this repository.

# Development and pull requests
To set up your development environment, please follow the steps below:

1. Clone the repository or a fork (for external contributors):
   ```bash
   git clone https://github.com/openvinotoolkit/geti-sdk.git
   cd geti-sdk
   ```
2. Checkout the desired branch or commit based off the `main` branch.
3. Install uv following the official [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

4. Set up the development environment with all dependencies:
   ```bash
   uv sync --all-extras
   ```

5. Make sure that `git lfs` is configured for your Git account, by following the
   steps [here](https://git-lfs.github.com/). Git LFS (Large File Storage) is used in
   this repo to manage certain data files used in the tests.

6. Run `git lfs pull` to download the test data.
7. Verify that the integration tests now run locally by executing `pytest tests/integration`
8. Set up the pre-commit hooks in the repo by running `pre-commit install`. Several pre-commit
   hooks are used in the repo to lint and format the code. The `pre-commit install`
   command sets up all of these hooks so that the checks will be performed on each
   commit you make.

You should now be ready to make changes, run the SDK integration tests and create a Pull Request!

## Testing your code
More details about the tests can be found in the [readme](tests/README.md) for the test suite.
If your changes require updating the tests or the test data, please refer to that document.
