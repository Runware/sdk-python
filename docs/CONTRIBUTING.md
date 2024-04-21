# Contributing to Runware SDK Python

Thank you for your interest in contributing to the Runware SDK Python project! We welcome contributions from the community to help improve and enhance the SDK.

## Getting Started

To get started with contributing, follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch for your changes: `git checkout -b my-feature-branch`.
4. Make your desired changes and commit them with descriptive commit messages.
5. Push your changes to your forked repository.
6. Open a pull request on the main repository's `main` branch.

## Guidelines

To ensure a smooth contribution process and maintain the quality of the project, please adhere to the following guidelines:

- Before starting to work on a new feature or bug fix, check the existing issues and pull requests to avoid duplication of effort.
- If you're planning to work on a significant change or new feature, it's recommended to open an issue first to discuss your proposal and gather feedback from the maintainers.
- Write clear, concise, and meaningful commit messages that describe the purpose of your changes.
- Make sure your code follows the project's coding style and conventions. We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- Include appropriate documentation for any new features, classes, or functions you add. Update existing documentation if necessary.
- Write unit tests to cover your changes and ensure they pass all existing tests. We use [pytest](https://docs.pytest.org/) for testing.
- Keep your pull requests focused and limited to a single feature or bug fix. If you have multiple unrelated changes, please submit separate pull requests.
- Be responsive to feedback and be open to making changes based on code reviews and discussions.

## Code of Conduct

Please note that this project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you are expected to uphold this code. Please report any unacceptable behavior to the project maintainers.

## Development Setup

To set up your development environment for the Runware SDK Python project, follow these steps:

1. Ensure you have Python 3.6 or above installed on your system.
2. Clone the repository and navigate to the project directory.
3. Create a virtual environment: `python -m venv .venv`.
4. Activate the virtual environment:
   - On macOS and Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
5. Install the project dependencies: `pip install -r requirements.txt`.
6. Install the pre-commit hooks: `pre-commit install`.
7. You're ready to start making changes!

## Testing

To run the test suite for the Runware SDK Python project, use the following command:

```bash
pytest tests/
```
