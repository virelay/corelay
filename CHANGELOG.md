# Changelog

## v0.3.0

*Release date to be determined.*

### General (v0.3.0)

- Added this changelog, as well as a contributors list, which contains a list of all people that made contributions to the project.
- Added a CSpell configuration for spell-checking the contents of the repository, checked all files, and corrected all spelling mistakes.

## v0.2.1

*Released on June 21, 2022.*

### General (v0.2.1)

- Fixed a small typo in the read me by @p16i in #8.
- Clarified the maximum number of neighbors in the `SparseKNN` affinity by replacing the confusing `if`-`else` expression with the `min` function by @chr5tphr in #10.
- Added a new logo, which has a design similar to the logo of ViRelAy and added a link to the documentation, as well as some badges to the read me by @chr5tphr in #12.

### CI/CD (v0.2.1)

- Added a GitHub Actions workflow, which runs the unit tests and builds the documentation by @chr5tphr in #13.
  - Fixed tests and PyLint errors
    - Fixed various PyLint style errors, mainly concerning the usage of f-strings instead of `format`.
    - Fixed the use of `sklearn.datasets.load_digits`, which now requires keyword-arguments.
    - Added `disable=unspecified-encoding` to the PyLint configuration file `pylintrc`, as this rule has a lot of trouble with false-positives.
  - Added a GitHub Actions workflow.
    - Defined the requirements for docs and tests (mostly) in the `setup.py` using `extras_require`.
    - Added a GitHub Actions workflow for tox.
    - Updated the `.readthedocs.yaml` configuration file to use the `extras_require` "docs" in the `setup.py`
    - Removed the `docs/requirements.txt`, which is no longer needed.
    - Added a "tests" badge to the read me.
    - Moved the documentation from imgmath to MathJax.
    - Moved the `corelay` package directory to `src/corelay` to fix the coverage report and prevent other pitfalls.
    - Change the linkcode URL in the documentation to correctly point to `src/corelay` on GitHub.
    - Updated the outdated PyLint configuration by removing deprecated disables.

### Documentation (v0.2.1)

- Added Sphinx-based documentation by @chr5tphr in #11.
  - Fixed the Docstrings and some NumPyDoc issues.
    - Removed trailing "s" after backticks.
    - Added missing docstring headers.
    - Used double backticks for inline-code.
  - Added Sphinx documentation.
    - Added an index page, with a very brief explanation, installation instructions, contents, and a reference to our paper.
    - Added an API reference with AutoSummary and AutoDoc.
    - Added a getting started section, which briefly and practically demonstrates Params, Processors, Tasks and Pipelines.
    - Added a bibliography with reference to our paper.
    - Added a favicon.svg, which currently is an "S", referring to the previous name of CoRelAy, which was Sprincl.
    - Ignored properties in AutoDoc to avoid doubling for now, since they are all documented as class attributes.
  - Added more Python versions to the tox configuration and integrated the building of the Sphinx documentation.
    - Added support for Python 3.7-3.9.
    - Moved flake8 to bottom, and only execute it in the Python 3.9 environment.
    - Reformated the PyTest command and added configuration for test coverage.
    - Added a new docs environment to build the Sphinx documentation.
    - Added a new coverage environment.
    - Made the PyLint command locations more explicit and only execute it in the Python 3.9 environment.
    - Added a PyTest configuration.
    - Added configuration the configuration for test coverage.
  - Added the configuration for Read the Docs: `.readthedocs.yaml`, which includes the build setup required for publishing the documentation to `readthedocs.org`.
- Remove outdated `requirements.txt`, which was supposed to be removed in #13 by @chr5tphr in #14.
- Remove "Sprincl" references, which was still used in two files, although the project has been renamed to CoRelAy by @chr5tphr in #15.

## v0.2.0

*Released on June 29, 2021.*

- Added a reference to the ClArC paper with a link and BibTeX in the read me by @chr5tphr in #1.
- Fixed submodules not being found and extra install by @chr5tphr in #2.
  - Installing `extra_require` packages must be installed somewhat differently than was shown in the read me, this was updated. Submodules of CoRelAy were also not found when directly installing from an URL. Using Setuptools `find_packages` with a specified include fixed this.
- Example fixes by @sebastian-lapuschkin in #3.
  - The imports from `sprincl` were fixed to imports from `corelay`.
  - SciKit-Image was updated and the usage of the `compare` function from `skimage.measure` was updated to the `structural_similarity` from `skimage.metrics`.
- Added a perplexity parameter to TSNE by @sebastian-lapuschkin in #4.
  - `TSNE` now has an optional perplexity parameter.
  - The default value is set to the same default of that `sklearn.manifold.TSNE` has, which is 30.
  - Lower perplexity means more focus on local structures, whereas higher perplexity means higher consideration of global structures.
- Updated the paper in the read me from the old paper to the new paper by @lecode-official in #5.
- Added LGPLv3+ as the license of the project by @chr5tphr in #6.
- Prepared the project for upload to PyPI by @chr5tphr in #7.
