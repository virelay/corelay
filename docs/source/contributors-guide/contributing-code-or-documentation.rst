==================================
Contributing Code or Documentation
==================================

Before contributing code or documentation, please ensure that there is no existing issue that aligns with your ideas. This helps avoid duplication of effort and ensures a seamless integration of your contributions into the project. If you have an idea for a contribution but can't find a related issue, we encourage you to open an issue, where you outline your proposal, before starting your work. This facilitates alignment with the rest of the team and may prevent unnecessary work.

To contribute code or documentation, follow these steps:

1. Fork the Repository
======================

To begin contributing to this project, you will first need to `fork the repository on GitHub <https://github.com/virelay/corelay/fork>`_. This creates a copy of the original repository under your own account. After forking the repository, it can be cloned using the following command:

.. code-block:: console

    $ git clone https://github.com/<your-username>/corelay.git

CoRelAy leverages `uv <https://github.com/astral-sh/uv>`_, a Python package and project manager, to streamline its development lifecycle. For detailed instructions on installing and utilizing ``uv``, please refer to its `comprehensive documentation <https://docs.astral.sh/uv/>`_. This tool enables the efficient installation of supported Python versions, management of virtual environments, handling of runtime and development dependencies, and project building.

Following the installation of ``uv``, please proceed to install the supported Python versions, which comprise 3.11, 3.12, and 3.13, as specified in the :repo:`source/.python-versions` file. This can be accomplished via the following command:

.. code-block:: console

    $ uv --directory source python install

Upon installing the supported Python versions, dependencies must be installed via the ``uv sync`` command. These are specified within the :repo:`source/pyproject.toml` project configuration file and constrained to specific versions in the :repo:`source/uv.lock` lock file. To install these dependencies, execute the following command:

.. code-block:: console

    $ uv --directory source sync --all-extras

To start your work, please create a new branch specifically for your feature or bug fix. When naming your branch, we recommend using kebab-case (lowercase words separated by hyphens) to clearly describe its purpose, e.g., ``my-new-feature``.

2. Make your Changes
====================

Before making any changes to the codebase, please familiarize yourself with the project by reading the :doc:`getting started guide <../getting-started/index>`, which will help you understand how the project works. To contribute to this project, please make the necessary changes to the codebase while adhering to our established coding standards and guidelines. We recommend extensively documenting your code with comments. We follow the `Google-style Docstring <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ convention.

When modifying existing code, please refrain from altering the project's coding style unless absolutely necessary. Changes to the code style can lead to unnecessary effort and frequent updates to accommodate varying preferences. To ensure consistency and maintainability, we utilize various tools to enforce our coding conventions, detect potential issues, prevent bugs, and statically type-check CoRelAy:

* `PyLint <https://www.pylint.org/>`_ -- Enforces coding style and detects potential issues and bugs.
* `PyCodeStyle <https://pycodestyle.pycqa.org/en/latest/intro.html>`_ -- Verifies code style consistency.
* `PyDocLint <https://jsh9.github.io/pydoclint/>`_ -- Ensures proper documentation and adherence to Docstring style guidelines.
* `MyPy <https://mypy-lang.org/>`_ -- Statically type-checks the code.

Before committing your changes, please verify that none of these tools produce warnings. Additionally, we advise against modifying the configuration of these tools unless a compelling reason exists (please provide details on your reasoning in the accompanying issue or pull request). For more information, please refer to the :ref:`testing-and-linting` section.

3. Write Unit Tests
===================

To ensure the reliability and stability of our codebase, please write comprehensive unit tests for the features you have added. Our goal is to achieve 100% test coverage for CoRelAy. This means that every line of code should be executed at least once during testing. Before committing your changes, please not only ensure that all unit tests pass without errors or failures, but also write sensible unit tests for all changes that you have made. This ensures that our codebase remains robust and maintains its expected functionality.

4. Update the Documentation
===========================

If your changes have impacted how the project is used or you made changes to its functionality, please ensure that the relevant sections of our documentation are updated accordingly. We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate our documentation, which can be found in the :repo:`docs/source` directory.

A local build of the documentation can be created using the following command:

.. code-block:: console

    $ uv --directory source run tox -e docs

.. _testing-and-linting:

5. Testing & Linting
====================

We use ``tox`` to run unit tests, linters and static type checkers on CoRelAy, as well as to build the documentation. If you've made any changes to CoRelAy or the documentation that require updates to configurations of the linters, type checker, or ``tox``, please ensure that the relevant sections in the following configuration files are are revised accordingly:

* **tox** -- :repo:`source/tox.ini`
* **PyLint** -- :repo:`tests/linters/.pylintrc`
* **PyCodeStyle** -- :repo:`tests/linters/.pycodestyle`
* **PyDocLint** -- :repo:`tests/linters/.pydoclint.toml`
* **MyPy** -- :repo:`tests/linters/.mypy.ini`

To run tests and build the documentation locally using ``tox``, execute the following command from the project root:

.. code-block:: console

    $ uv --directory source run tox run

Unit tests are run on all supported Python versions (3.11, 3.12, and 3.13). They can be run individually using the following command:

.. code-block:: console

    $ uv --directory source run tox -e py311
    $ uv --directory source run tox -e py312
    $ uv --directory source run tox -e py313

To generate an HTML coverage report, you can add the ``coverage`` environment to the list of environments to run:

.. code-block:: console

    $ uv --directory source run tox -e py313,coverage

The linters and static type checkers can also be run individually using the following commands:

.. code-block:: console

    $ uv --directory source run tox -e pylint
    $ uv --directory source run tox -e pycodestyle
    $ uv --directory source run tox -e pydoclint
    $ uv --directory source run tox -e mypy

Finally, we use a Markdown linter to ensure the quality of the read me and a spell checker to verify the correct spelling of all text, including code files. Both of these tools are based on `Node.js <https://nodejs.org>`_. If you do not have Node.js and NPM installed, this can be easily achieved using the `Node Version Manager (nvm) <https://github.com/nvm-sh/nvm>`_. We recommend installing an `active LTS or maintenance LTS release <https://nodejs.org/en/about/releases/>`_ of Node.js. Once Node.js and NPM are installed, you can install the Markdown linter and the spell checker using the following commands:

.. code-block:: console

    $ npm --prefix tests/linters/markdownlint install
    $ npm --prefix tests/linters/cspell install

The Markdown linter and the spell checker can be run using the following commands:

.. code-block:: console

    $ npm --prefix tests/linters/markdownlint run markdownlint
    $ npm --prefix tests/linters/cspell run cspell

If your changes require updates to the configurations of the Markdown linter or the spell checker, please update the following configuration files:

* **Markdown Linter** -- :repo:`tests/linters/markdownlint/.markdownlint.yaml`
* **Spell Checker** -- :repo:`tests/linters/cspell/.cspell.json`

Our continuous integration and deployment (CI/CD) pipeline is built using GitHub Actions Workflows. You can use the `act tool <https://nektosact.com/>`_ to test the GitHub Actions workflow locally. Install the act tool according to the `official installation instructions <https://nektosact.com/installation/index.html>`_. After the installation, the GitHub Actions workflow can be run locally using the following commands:

.. code-block:: console

    $ act                # Runs all workflows
    $ act --job <job-id> # Runs a single job with the specified ID (e.g., unit-tests, build-documentation, pylint, etc.)

When prompted to select a Docker image, we recommend using the "full" image.

If your changes require updates to the GitHub Actions workflows, please update the following configuration files:

* **Unit Tests, Linting & Building** -- :repo:`.github/workflows/tests.yml`
* **Deployment to PyPI** -- :repo:`.github/workflows/deploy.yml`

To ensure a successful review of your pull request, please verify that:

* All linters and static type checkers pass without errors.
* Unit tests succeed for all supported Python versions (3.10 - 3.13).
* The documentation builds successfully.

If any of these checks fail, we will not be able to accept the pull request.

6. Update the Changelog
=======================

As part of your contribution, please ensure that the project's changelog is updated to reflect the modifications you've made. This can be done by editing the :repo:`CHANGELOG.md` file.

By recording your changes in our changelog, we can maintain a clear and accurate history of updates, making it easier for users and developers to track progress and understand the impact of each release.

7. Add Yourself to the Contributors List
========================================

As a final step before committing your changes and making a pull request, please consider adding your name to our contributors list in :repo:`CONTRIBUTORS.md`. This allows us to formally recognize and appreciate your contribution to the project.

You may choose to add yourself under a pseudonym or use your actual name; we respect your preference and encourage you to acknowledge your hard work in making this project better.

8. Commit Your Changes
======================

To ensure that your contributions are easily reviewable and maintainable, please strive for a few meaningful, coherent commits with descriptive commit messages. We follow the conventional 50/72 rule:

* A brief subject line (not exceeding 50 characters) that summarizes the changes.
* A detailed description (with lines capped at 72 characters), separated from the subject line by a blank line.

Additionally, after each commit, please ensure that the repository remains in a healthy state. If the ``main`` branch has progressed since you branched it off, use a Git rebase instead of a merge to avoid unnecessary merge commits. This helps keep the commit history clean and makes it easier for others to review your changes.

9. Submit Your Contribution for Review
======================================

Once you've completed your development work, push your changes to your forked repository and create a pull request against the main repository. When creating the pull request, please provide a clear and detailed description of your changes, including how they address the specific issue or feature being implemented.

Be sure to reference the relevant issues in your description so that our review team can easily identify the context for your contribution. We'll strive to review your submission as soon as possible, providing feedback and guidance to ensure a smooth integration process.
