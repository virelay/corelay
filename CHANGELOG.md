# Changelog

## v1.0.0

*Released on July 21, 2025.*

### General Updates in v1.0.0

- Renamed the `master` branch to `main` in order to avoid any links to sensitive topics. All references to the `master` branch in the repository were updated to `main`.
- Added this changelog, as well as a contributors list, which contains a list of all people that made contributions to the project.
- CSpell, a spell-checker, was added to the project. The contents of the repository, including the source code, were spell-checked and all spelling mistakes were corrected.
- Markdownlint, a linter for Markdown documents, was added to the project and all Markdown files in the project were updated to follow the MarkdownLint style guide.
- Moved the logo and its source file from the docs to a separate top-level `design` directory:
  - The source file was cleaned up:
    - Converted the title of the logo to a path, because the font is from Google Fonts and not available in the SVG. It would be possible to embed the font, but this would increase the size of the SVG significantly and would require us to include the license.
    - The logo was previously only available with the title. For this reason a second page was added to the SVG, which contains the logo without the title.
    - Named and cleaned up all objects and groups in the SVG.
  - A PNG version of the logo without the title was also added.
  - All references to the old logo were updated to point to the new location. The URL used in the read me was made absolute, because the read me is also used for the PyPI package and PyPI would not be able to resolve the relative URL to the logo on GitHub.
- The project is dual-licensed under the GNU General Public License Version 3 (GPL-3.0) or later, and the GNU Lesser General Public License Version 3 (LGPL-3.0) or later. The GPL-3.0 license is in the `COPYING` file and the LGPL-3.0 license is in the `COPYING.LESSER` file. Additionally, there used to be a `LICENSE` file, which contained a note about the dual-licensing. This was, however, confusing, as GitHub does not recognize that the file is only a note about the dual-licensing and not the actual license. The `LICENSE` file was removed and the note about the dual-licensing was added to the read me.
- Added a `CITATION.cff` file, which contains the necessary information to cite this repository. This file is based on the [Citation File Format (CFF)](https://citation-file-format.github.io) standard. This file is supported by GitHub and results in a "Cite this repository" button on the website, which allows users to directly generate a proper citation for the repository in multiple different formats.

### CoRelAy Updates in v1.0.0

- Converted the CoRelAy project from a `setup.py` project to a uv project:
  - A `pyproject.toml` file was created, which is configured to do the same as the `setup.py` file.
  - The tox configuration was updated and now uses tox-uv to run all commands via uv instead of directly creating environments. This means, that all Python environments can now be run without having to install multiple Python versions.
  - The tox configuration was also cleaned up.
  - Support for Python 3.7, 3.8, and 3.9 were removed. Python 3.7 and 3.8 have already reached their end-of-life, and Python 3.9 is about to reach its end-of-life and already only receives security updates. Besides their support status, some of the dependencies (especially tox-uv) do not support them anymore. Even Python 3.10 is already unsupported by some of the dependencies. For this reason, only Python 3.11, 3.12, and 3.13 are supported now. They are recorded in the `.python-versions` file, which makes it trivial to install them using uv.
- Updated the Python dependencies:
  - The Python dependencies specified in the `pyproject.toml` file were updated to their respective latest versions.
- Updated the unit tests:
  - Some of the unit tests were failing due to changes in the dependencies, which were updated to fix the issues:
    - Since NumPy 1.26, NumPy array functions are no longer actual functions, instead they are now wrapped in a class, which calls into the C implementation of the corresponding functions. Unfortunately, this class does not implement `FunctionType`. Many of the classes in CoRelAy allow a `dtype`, which is checked for consistency with the input data. For example, the `Param` class allows to specify a `dtype` and a default value. The default value is checked for consistency with the `dtype` and an exception is raised if the default value is not of the correct type. Since NumPy array functions are no longer actual functions, code like `pooling_function: Annotated[FunctionType, Param(FunctionType, numpy.sum)]` cannot be used anymore. In fact, this code would also not work for builtin functions like `sum`, for class methods, and some other cases. For this reason, the consistency check will now detect if the user specified `FunctionType` as the `dtype` and will automatically add the following types to the list of accepted types: `BuiltinFunctionType`, `BuiltinMethodType`, `MethodType`, `numpy.ufunc`, and `type(numpy.max)` (the type of NumPy array functions is private). This will make old code work as expected and open up new possibilities for the user, like using builtin functions and class methods.
    - Functions in SciKit Learn and SciKit Image that allow single-channel or multi-channel images used to have a boolean `multichannel` constructor parameter. This parameter was deprecated in favor of specifying the axis of the channels using a new `channel_axis` parameter. A value of `None` indicates that the image is single-channel. The usage of `multichannel` was removed and replaced with the appropriate `channel_axis` parameter.
    - The SciKit Learn implementation of the t-SNE dimensionality reduction algorithm, represented by the `TSNE` class, now uses PCA as the default initialization method instead of a random initialization. The `precomputed` metric is not compatible with PCA initialization, which was used in CoRelAy, which caused an exception to be raised. For this reason, the initialization method is now explicitly set to `random` in the `TSNE` class.
    - The `AgglomerativeClustering` class in SciKit Learn now uses the `metric` constructor parameter instead of `affinity` to specify the distance metric. Uses of the `affinity` parameter were replaced with the `metric` parameter.
  - Brought the coverage of the unit tests to 100%.
  - The unit tests were moved from the `tests` folder to `tests/unit_tests`, which clears up space for other test files.
- Updated the linting of the project:
  - First of all, the Flake8 linter was removed. It is a wrapper for the PyFlakes and PyCodeStyle linters, and Ned Batchelder's McCabe script, which is used to compute the McCabe complexity. PyFlakes is a static code analyzer similar to PyLint, but way less useful. McCabe complexity is a useful metric, but it is not currently used by the project. For this reason, the Flake8 linter was removed and PyCodeStyle is now directly used instead.
  - Also, MyPy, a static type checker for Python, and PyDocLint, a docstring linter, were added to the project. The configuration for PyLint was updated by removing all options that only have their default values, as they are automatically set by PyLint. The remaining options were updated to make them less restrictive for the project. Mainly, maximum and minimum values were increased/decreased to allow for more flexibility.
  - Flake8 was removed from the tox configuration and the GitHub Actions workflow, and the new linters were added.
  - The configurations for the linters now reside in the `tests/linters` directory. The existing PyLint configuration was moved there.
  - All errors and warnings from the linters were fixed. In particular, the following incomplete list of changes were made:
    - The imports were sorted in all Python files. They are now categorized by standard library imports, third-party library imports, and local imports, each separated by a blank line. Each category is now sub-categorized into regular imports and "from-imports". These are not separated by blank lines. Each sub-category is sorted alphabetically.
  - The docstrings were updated to now follow the Google style guide, which is easier to read and write than the NumPyDoc style. Also, this style is supported by PyDocLint, which is now used to lint the docstrings.
  - Missing module docstrings, function docstrings, class docstrings, and method docstrings were added to all Python files.
  - The maximum line length was set to 150 characters, which is a bit less restrictive than the previous 120 characters. Still, this should fit two files side by side on modern high-resolution monitors.
  - `Click` was removed as the command line argument parser. It is a great library, but it has the disadvantage of producing a lot of PyLint errors. Instead, the built-in `argparse` library is now used, which is a bit more verbose, but it does not produce any PyLint errors.
  - Variables were renamed to be more descriptive. Previously, most variable names were heavily abbreviated and some of them were not very intuitive.
  - Most of the inline PyLint disables were removed and either the offending rule was directly disabled or the code was changed to not trigger the rule anymore. This was done to make the code cleaner and easier to read. For all remaining inline PyLint disables, the reason for the disable was better explained.
  - Relative imports were replaced by absolute imports, because they make it easier to understand where the module is located. Especially because the project has multiple modules that are named the same in different sub-packages.
  - Type hints were added to all functions, methods, class attributes, and variables where necessary. Also, parameters and attributes of type string that had a specific set of possible values are now typed using the `Literal` type. This makes it possible for MyPy to check if specified values are valid.
  - In the unit tests, the fixture parameters were masking the fixture function names. This was fixed by renaming the fixture function names `get_<fixture_name>_fixture` and specifying an explicit fixture name. The fixtures are now also explicitly scoped to the module level, although this did not cause any problems previously.
- The `corelay/version.py` file was deleted as it is automatically generated during the build process and should not be checked into source control.
- In general, the code was cleaned up to make it easier to read, understand, and maintain. Some instances of dead code were eliminated. The goal was to make the code more Pythonic and to follow the PEP 8 style guide as closely as possible, while still maintaining backwards compatibility. This was, however, not possible in all cases and some backwards compatibility was sacrificed for the sake of improved static typing. This includes:
  - Some meta classes were removed and replaced by protocols (which are implicit interfaces).
  - The most important change was how slots are defined:
    - Previously, the slots were defined by assigning an instance of the `Slot` class (e.g., using the `Param` class, which derives from the `Slot` class) to a class attribute.
    - The problem with this is, that Python allows users to access class attributes via the class name or via an instance of the class, e.g., using `self`. This is unless an instance attribute with the same name is defined is defined, in which case the class attribute is accessed when using the class name and the instance attribute is accessed when using the class instance.
    - The previous version of CoRelAy exploited this by returning the `Slot` instance when accessing the attribute via the class name and returning the value of the `Slot` when accessing the attribute via the class instance.
    - Unfortunately, MyPy cannot handle this and therefore always expects the `Slot` instance to be returned when accessing the attribute in any way and thus raises an error because there is a type mismatch.
    - This was fixed by introducing a new syntax, where slots are now defined via type hints.
    - Slots are now defined by declaring a class attribute with a type hint using the `Annotated` type. This is a type that allows to specify a type and additional metadata. For example, a string `Param` can be defined as follows: `param: Annotated[str, Param(str, 'Default value')]`.
    - This way, MyPy can infer the runtime type of the attribute and the metadata is used to define the `Slot`.
    - The old syntax is still supported, but it is no longer recommended and may be removed in the future.
    - A deprecation warning is raised when using the old syntax, so that users can easily find and fix the code that uses the old syntax.
- Fixed a bug in the `RadialBasisFunction` affinity processor: The formula for the Radial Basis Function (RBF) kernel was incorrect. The distance matrix was not squared resulting in the wrong formula $K(d) = e^{-\frac{d}{2\sigma^2}}$ instead of the correct formula $K(d) = e^{-\frac{d^2}{2\sigma^2}}$. This was fixed by squaring the distance matrix.
- Fixed a bug in the `Histogram` processor: The processor was using the `numpy.histogramdd` function, which computes a multivariate histogram, but the processor was meant to compute a histogram over the channels of the input data, for which the `numpy.histogramdd` function is not suitable. Instead, the `numpy.histogram` function is now used in conjunction with the `numpy.stack` function to compute the histogram over the channels of the input data. Also, the `Histogram` processor was not able to deal with channel-last data, which is now supported. The `Histogram` processor created in the `virelay_analysis.py` example script was also not working as expected: Although it was using the `numpy.histogram` function, it did not account for its return type, which is a tuple containing the histogram and the bin edges. Since we now have a working implementation of the `Histogram` processor, the example script was updated to use it.
- The `Shaper` flow processor was extended to support dictionaries and string indices. It seems like, these features were already expected to be present, but they were prevented by some minor mistakes: Dictionaries were not supported, because the `Shaper` processor tested for the type of the input data by testing if they implemented the `Sequence` protocol, which is not the case for dictionaries. Instead, dictionaries implement the `Mapping` protocol, which is now also tested for. String indices were not supported, because the indices were tested for implementing the `Sequence` protocol, which is also the case for strings. This meant, that each character of a string was treated as a separate index, which is not the intended behavior. Now, it is also checked if the indices are not strings.

### CI/CD Updates in v1.0.0

- The GitHub Actions workflow was updated to point to the new locations of the source code, unit tests, and configuration files.
- Converted the GitHub Actions workflow to use uv to run the tests, linters, and build the documentation.
- The GitHub Actions workflow was cleaned up and documented.
- Split up the GitHub Actions workflow job that ran the linters into separate jobs, which allows for better parallelization and faster execution of the workflow.
- Updated the `actions/checkout` action used in the GitHub Actions workflow from version 2 to version 4.
- Removed the GitHub Actions workflow matrix configurations for Python 3.7, as it is no longer supported by the project.
- Added a job to the GitHub Actions workflow, which spell-checks the repository.
- The GitHub Actions workflow now runs on pull requests and merges to the `main` and the `develop` branches. The workflow was previously only ran on pull requests and merges to the `main` branch. This was changed, because every feature branch that is to be merged into `develop` should be tested and linted before it is merged. Otherwise, build, test, and linting errors would only be detected just before the release of a new version, when the `develop` branch is merged into `main`.
- Added a new GitHub Actions workflow, which builds the project and publishes it to PyPI. This workflow is triggered when GitHub release for a new version is created.
- The configuration for the GitLab CI, which was stored in the `.gitlab-ci.yml` file, was removed. The project is no longer being hosted on GitLab, and the CI configuration is no longer needed.

### Documentation Updates in v1.0.0

- The documentation was largely extended and a bit reorganized:
  - The API reference documentation was moved from the `reference` directory to the `api-reference` directory. This was done to better reflect the purpose of the documentation, since there are also bibliographical references in the documentation.
  - The "Getting Started" section was expanded to new section with multiple sub-sections: "Installation", "Basic Usage", and an "Example Project".
    - The installation and basic usage sections were moved from the original "Getting Started" section.
    - The "Example Project" section contains a more elaborate example of how to use CoRelAy to analyze a dataset generated by Zennit using the SpRAy workflow that can be visualized using ViRelAy.
  - A new "Contributor's Guide" section was added with sub-sections on how to report issues and feature requests, and how to contribute code or documentation.
  - A new "Migration Guide" section was added to help users migrate from CoRelAy v0.2 to CoRelAy v1.0. This was done, because the changes made in CoRelAy v1.0 are quite extensive and it is not easy to find out what has changed and how to adapt existing code to the new version.
  - More citations were added to the documentation to relevant literature.
- The configuration for Read the Docs was updated to use the latest available versions of Ubuntu (24.04) and Python (3.12). It was also documented.
- The configuration for Sphinx was updated:
  - The `sphinxcontrib.datatemplates` extension was removed, because it was not used in the documentation.
  - Added the `sphinx.ext.intersphinx` extension to add links in the documentation to external documentations, e.g., the Python standard library documentation and the NumPy documentation.
  - The docstrings in the CoRelAy source code now use `:py:*:` directives to link references to classes, functions, methods, modules, and values to their respective external documentation.
  - All warnings and errors that were reported by the linters were fixed and the configuration generally cleaned up to make it more readable and maintainable.
  - The `docs` environment in the tox configuration now uses Python 3.13.3, which causes an error during the building of the documentation, because the `pkg_resources` module was deprecated and removed in Python 3.12. This module was used to get the source code file paths and line numbers of functions, classes, methods, etc. for the `linkcode` extension. This was replaced by the `inspect` module.
  - The invocation of Sphinx in tox to build the documentation was updated to include the `--fresh-env` option, so that the documentation is always built in a fresh environment. This solves some issues with the documentation build, where the documentation was not updated correctly after changes were made to the source code or the documentation itself.
  - The correct capitalization of the project name is now specified in the Sphinx configuration and the year in the copyright is now always set to the current year instead of hard-coding it.
- The example scripts were updated to reflect the coding style and docstring conventions used in the CoRelAy library. They were moved from the `example` directory to the `docs/examples` directory. This was done because the examples are part of the documentation and to keep the root directory clean and free of unnecessary clutter.
- A custom CSS file was added to the documentation to change the alignment of the documentation text to be justified and to automatically break long words. This was done to improve the readability of the documentation and to align it with usual scientific documents.
- The width of the labels in the bibliography was set to a fixed size, so that the labels all have the same width. This was done, because they looked inconsistent and messy before.
- The logo was added to the documentation, which previously contained a copy of the logo in the `docs/images` directory, but did not include it. The version contained in the `docs/images` directory was removed and the index page now directly references the logo in the `design` directory.
- The favicon used in the documentation was updated to use the SVG version of the logo without the title. Previously, it was still the old "S" logo, which was from before CoRelAy was renamed from Sprincl.
- One of the new dependency versions (`metrohash-python`) requires a C++ compiler to build and uses the `c++` command, which may not be available on all systems. To ensure that users are not confused by this, a note was added to the read me file explaining that the `c++` command is required to build the project and showing how to install it on Fedora (one of the systems that do not have the `c++` command installed by default).
- The read me of the repository was also updated. The introduction was extended to better explain the CoRelAy library. A section on the features of CoRelAy was added. The installation instructions and the usage section were now wrapped in a new "Getting Started" section. The example code in the usage section was updated to reflect the changes made to the example scripts. The "Contributing" section was also updated to include a proper description and links to the contributors guide in the documentation.

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
  - SciKit-Image was updated and the usage of the `compare` function from `skimage.measure` was updated to the `structural_similarity` function from `skimage.metrics`.
- Updated the paper in the read me from the old paper to the new paper.
