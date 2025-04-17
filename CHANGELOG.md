# Changelog

## v0.3.0

*Release date to be determined.*

### General Updates in v0.3.0

- Renamed the `master` branch to `main` in order to avoid any links to sensitive topics. All references to the `master` branch in the repository were updated to `main`.
- Added this changelog, as well as a contributors list, which contains a list of all people that made contributions to the project.
- Added a CSpell configuration for spell-checking the contents of the repository, checked all files, and corrected all spelling mistakes.
- Moved the logo and its source file from the docs to a separate top-level `design` directory:
  - The source file was cleaned up:
    - Converted the title of the logo to a path, because the font is from Google Fonts and not available in the SVG. It would be possible to embed the font, but this would increase the size of the SVG significantly and would require us to include the license.
    - The logo was previously only available with the title. For this reason a second page was added to the SVG, which contains the logo without the title.
    - Named and cleaned up all objects and groups in the SVG.
  - A PNG version of the logo without the title was also added.
  - All references to the old logo were updated to point to the new location. The URL used in the read me was made absolute, because the read me is also used for the PyPI package and PyPI would not be able to resolve the relative URL to the logo on GitHub.
- The project is dual-licensed under the GNU General Public License Version 3 (GPL-3.0) or later, and the GNU Lesser General Public License Version 3 (LGPL-3.0) or later. The GPL-3.0 license is in the `COPYING` file and the LGPL-3.0 license is in the `COPYING.LESSER` file. Additionally, there used to be a `LICENSE` file, which contained a note about the dual-licensing. This was, however, confusing, as GitHub does not recognize that the file is only a note about the dual-licensing and not the actual license. The `LICENSE` file was removed and the note about the dual-licensing was added to the read me.
- Added a `CITATION.cff` file, which contains the necessary information to cite this repository. This file is based on the [Citation File Format (CFF)](https://citation-file-format.github.io) standard. This file is supported by GitHub and results in a "Cite this repository" button on the website, which allows users to directly generate a proper citation for the repository in multiple different formats.
- The configuration for the GitLab CI, which was stored in the `.gitlab-ci.yml` file, was removed. The project is no longer being hosted on GitLab, and the CI configuration is no longer needed.

### CoRelAy Updates in v0.3.0

- Converted the CoRelAy project from a `setup.py` project to a uv project:
  - A `pyproject.toml` file was created, which is configured to do the same as the `setup.py` file.
  - The source code was moved from `src/corelay` to `source/corelay`.
  - The tox configuration was updated and now uses tox-uv to run all commands via uv instead of directly creating environments. This means, that all Python environments can now be run without having to install multiple Python versions.
  - The tox configuration was also cleaned up.
  - Support for Python 3.7 was removed, not only because it has already reached its end-of-life, but also because some of the dependencies (especially tox-uv) do not support it anymore. The two remaining supported Python versions (3.8 and 3.9) are now recorded in the `.python-versions` file, which makes it trivial to install them using uv.
- Updated the Python dependencies in the `pyproject.toml` to their respective latest versions.
- Some of the new dependency versions no longer support Python 3.8 and Python 3.9 (as well as Python 3.10). For this reason, the project was migrated to support Python 3.11, 3.12, and 3.13. The `.python-versions` file, the tox configuration, and the GitHub Actions workflow were updated to reflect this change.
- The unit tests were moved from the `tests` folder to `tests/unit_tests`, which clears up space for other test files.
- The tox configuration was moved from the root directory to the `tests` directory, which is more appropriate, as tox is mostly used for testing and linting.
- The configurations for the linters PyLint and Flake8 were moved from the root directory of the repository (in the case of PyLint) and from the tox configuration file (in the case of Flake8) into the `tests/linters` directory. The CSpell configuration was also moved there.
- The `corelay/version.py` file was deleted as it is automatically generated during the build process and should not be checked into source control.

### CI/CD Updates in v0.3.0

- The GitHub Actions workflow was updated to point to the new locations of the source code, unit tests, and configuration files.
- Converted the GitHub Actions workflow to use uv to run the tests, linters, and build the documentation.
- The GitHub Actions workflow was cleaned up and documented.
- Split up the GitHub Actions workflow job that ran PyLint and Flake8 into two separate jobs, which allows for better parallelization and faster execution of the workflow.
- Updated the `actions/checkout` action used in the GitHub Actions workflow from version 2 to version 4.
- Removed the GitHub Actions workflow matrix configurations for Python 3.7, as it is no longer supported by the project.
- Added a job to the GitHub Actions workflow, which spell-checks the repository.

### Documentation Updates in v0.3.0

- The configuration for Read the Docs was updated to use the latest available versions of Ubuntu (24.04) and Python (3.12). It was also documented.
- The logo was added to the documentation, which previously contained a copy of the logo in the `docs/images` directory, but did not include it. The version contained in the `docs/images` directory was removed and the index page now directly references the logo in the `design` directory.
- The favicon used in the documentation was updated to use the SVG version of the logo without the title. Previously, it was still the old "S" logo, which was from before CoRelAy was renamed from Sprincl.
- One of the new dependency versions (`metrohash-python`) requires a C++ compiler to build and uses the `c++` command, which may not be available on all systems. To ensure that users are not confused by this, a note was added to the read me file explaining that the `c++` command is required to build the project and showing how to install it on Fedora (one of the systems that do not have the `c++` command installed by default).

## v0.2.1

*Released on June 21, 2022.*

### General Updates in v0.2.1

- Fixed a small typo in the read me.
- Added a new logo, which has a design similar to the logo of ViRelAy and added a link to the documentation, as well as some badges to the read me.

### CoRelAy Updates in v0.2.1

- Updated the outdated PyLint configuration by removing deprecated disables and fixed tests and PyLint errors:
  - Fixed various PyLint style errors, mainly concerning the usage of f-strings instead of `format`.
  - Fixed the use of `sklearn.datasets.load_digits`, which now requires keyword-arguments.
  - Added `disable=unspecified-encoding` to the PyLint configuration file `pylintrc`, as this rule has a lot of trouble with false-positives.
- Moved the `corelay` package directory to `src/corelay` to fix the coverage report and prevent other pitfalls.
- Clarified the maximum number of neighbors in the `SparseKNN` affinity by replacing the confusing `if`-`else` expression with the `min` function.
- Defined the requirements for docs and tests (mostly) in the `setup.py` using `extras_require` and removed the `docs/requirements.txt`, which is no longer needed.
- Fixed the Docstrings and some NumPyDoc issues:
  - Removed trailing "s" after backticks.
  - Added missing docstring headers.
  - Used double backticks for inline-code.
- Removed the outdated `requirements.txt`, which was supposed to be removed in #13.

### CI/CD Updates in v0.2.1

- Added a GitHub Actions workflow, which runs the unit tests and builds the documentation.
- Added more Python versions to the tox configuration and integrated the building of the Sphinx documentation:
  - Added support for Python 3.7-3.9.
  - Moved flake8 to bottom, and only execute it in the Python 3.9 environment.
  - Reformated the PyTest command and added configuration for test coverage.
  - Added a new docs environment to build the Sphinx documentation.
  - Added a new coverage environment.
  - Made the PyLint command locations more explicit and only execute it in the Python 3.9 environment.
  - Added a PyTest configuration.
  - Added the configuration for test coverage.

### Documentation Updates in v0.2.1

- Migrated the documentation from using imgmath to MathJax.
- Added a "tests" badge to the read me.
- Added Sphinx-based documentation:
  - Added an index page, with a very brief explanation, installation instructions, contents, and a reference to our paper.
  - Added an API reference with AutoSummary and AutoDoc.
  - Added a getting started section, which briefly and practically demonstrates Params, Processors, Tasks and Pipelines.
  - Added a bibliography with reference to our paper.
  - Added a favicon.svg, which currently is an "S", referring to the previous name of CoRelAy, which was Sprincl.
  - Ignored properties in AutoDoc to avoid doubling for now, since they are all documented as class attributes.
- Added the configuration for Read the Docs: `.readthedocs.yaml`, which includes the build setup required for publishing the documentation to `readthedocs.org`.
- Updated the `.readthedocs.yaml` configuration file to use the `extras_require` "docs" in the `setup.py`
- Changed the linkcode URL in the documentation to correctly point to `src/corelay` on GitHub.
- Removed "Sprincl" references, which was still used in two files, although the project has been renamed to CoRelAy.

## v0.2.0

*Released on June 29, 2021.*

### General Updates in v0.2.0

- Added LGPLv3+ as the license of the project.
- Prepared the project for upload to PyPI.

### CoRelAy Updates in v0.2.0

- Fixed submodules not being found and clarified how the `extra_require` dependencies have to be installed:
  - Submodules of CoRelAy were also not found when directly installing from an URL. Using Setuptools `find_packages` with a specified include fixed this.
  - Installing `extra_require` packages must be installed somewhat differently than was shown in the read me, this was updated.
- Added a perplexity parameter to TSNE:
  - `TSNE` now has an optional perplexity parameter.
  - The default value is set to the same default of that `sklearn.manifold.TSNE` has, which is 30.
  - Lower perplexity means more focus on local structures, whereas higher perplexity means higher consideration of global structures.

### Documentation Updates in v0.2.0

- Added a reference to the ClArC paper with a link and BibTeX in the read me.
- Fixed some minor problems in the examples:
  - The imports from `sprincl` were fixed to imports from `corelay`.
  - SciKit-Image was updated and the usage of the `compare` function from `skimage.measure` was updated to the `structural_similarity` from `skimage.metrics`.
- Updated the paper in the read me from the old paper to the new paper.
