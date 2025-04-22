"""Represents the configuration for Sphinx, which is executed at build time and can be used to configure Sphinx's input and output behavior."""

# pylint: disable=invalid-name

import inspect
import os
import re
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime
from subprocess import run, CalledProcessError
from types import ModuleType
from typing import Any, Literal

from pybtex.database import Entry
from pybtex.plugin import register_plugin
from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle
from sphinx.application import Sphinx


LanguageDomain = Literal['py', 'c', 'cpp', 'javascript']
"""Represents the different language domains for which the linkcode extension can generate links to the source in the repository."""


class AuthorYearLabelStyle(BaseLabelStyle):
    """Represents a citation label style, which uses the format "[<first-author> et al., <year>]"."""

    def format_labels(self, sorted_entries: list[Entry]) -> Iterator[str]:
        """Formats the labels of all bibliography entries.

        Args:
            sorted_entries (list[Entry]): The sorted list of bibliography entries.

        Yields:
            str: Yields the formatted labels of all bibliography entries.
        """

        for entry in sorted_entries:
            yield f'[{entry.persons["author"][0].last_names[0]} et al., {entry.fields["year"]}]'


class AuthorYearStyle(PlainStyle):
    """Represents a custom bibliography style, which uses the citation label format "[<first-author> et al., <year>]"."""

    default_label_style: str | type = AuthorYearLabelStyle
    """The label style to use for the bibliography entries."""


def get_latest_git_tag() -> str:
    """Retrieves the latest Git tag in the source code repository.

    Returns:
        str: Returns the name of the latest Git tag in the source code repository. If no tags are available, then "main" is returned.
    """

    # Tries to get the most recent tag in the source code repository using the git describe command, which returns the closest tag that can be reached
    # from the specified revision, which in this case is the latest commit on main, if no tags are available, then "main" is returned as a fallback
    try:
        return run(
            ['git', 'describe', '--tags', 'HEAD'],
            capture_output=True,
            check=True,
            text=True
        ).stdout[:-1]
    except CalledProcessError:
        return 'main'


def get_object_by_name(name: str, module: ModuleType) -> ModuleType | type[Any] | None:
    """Retrieves the object with the specified name from the specified module. The object can be a module, a class, a method, a function, or a
    variable. If the name is not valid or the object does not exist, then None is returned.

    Args:
        name (str): The name of the object to retrieve.
        module (ModuleType): The module from which to retrieve the object.

    Returns:
        ModuleType | type[Any] | None: Returns the object with the specified name from the specified module. If the object does not exist, then
            None is returned.
    """

    name_components = name.split('.')
    if not name_components:
        return None

    object_to_retrieve: ModuleType | type[Any] = module
    for name_component in name_components:
        try:
            object_to_retrieve = getattr(object_to_retrieve, name_component)
        except AttributeError:
            return None

    return object_to_retrieve


def get_top_level_module(module_name_hierarchy: str) -> ModuleType | None:
    """Retrieves the top-level module from the specified module name hierarchy. For example, if the module hierarchy is `a.b.c`, then the module `a`
    will be returned. If the module name hierarchy is `a`, then `a` will be returned. If the module name hierarchy is not valid or the top-level
    module of the hierarchy does not exist, then None is returned.

    Args:
        module_name_hierarchy (str): The module name hierarchy from which the top-level module is to be retrieved.

    Returns:
        ModuleType | None: Returns the top-level module from the specified module name. If the module does not exist, then None is returned.
    """

    module_hierarchy_names = module_name_hierarchy.split('.')
    if not module_hierarchy_names:
        return None

    top_module_name = module_hierarchy_names[0]
    top_module = sys.modules.get(top_module_name)
    if top_module is None:
        return None

    return top_module


def is_type_alias_for_literal(object_to_check: Any) -> bool:
    """Checks if the specified object is a type alias for a Literal[...] type.

    Args:
        object_to_check (Any): The object to check.

    Returns:
        bool: Returns True if the specified object is a type alias for a Literal[...] type, otherwise False.
    """

    return type(object_to_check).__name__ == '_LiteralGenericAlias'


def get_source_code_location_of_literal_type_alias(name: str, module: ModuleType) -> tuple[str | None, int | None, int | None]:
    """Retrieves the path to the source code file that contains the type alias for the Literal[...] type, as well as the start and end line numbers of
    the type alias declaration. If the type alias for the Literal[...] type does not exist, then None is returned for all three values.

    Args:
        name (str): The name of the type alias for the Literal[...] type.
        module (ModuleType): The module that contains the type alias for the Literal[...] declaration.

    Returns:
        tuple[str | None, int | None, int | None]: Returns a tuple containing the path to the source code file that contains the type alias for the
            Literal[...] type, as well as the start and end line numbers of the type alias declaration. If the type alias for the Literal[...] type
            does not exist, then None is returned for all three values.
    """

    # Checks if the module's file path is available, if not, then there is no source file that we can check
    if module.__file__ is None:
        return None, None, None

    # Loads the module's source code file into memory
    with open(module.__file__, mode='r', encoding='utf-8') as module_file:
        module_file_contents = module_file.read()

    # Checks if the type alias declaration is present in the module's source code file, if not then we cannot generate a URL to the source code
    match: re.Match[str] | None = re.search(
        r'(type \s*)?' + name + r'\s* (: \s* TypeAlias)? \s* = \s* Literal \s* \[ \s* (\s* [\'"] [^\'"]* [\'"] ,?)* \s* \]',
        module_file_contents,
        re.VERBOSE
    )
    if not match:
        return None, None, None

    # Gets the start and the end line number of the type alias declaration
    start_index, end_index = match.span()
    start_line_number = module_file_contents.count('\n', 0, start_index) + 1
    end_line_number = module_file_contents.count('\n', 0, end_index) + 1

    # Returns the path to the source code file that contains the type alias declaration, and the start and end line numbers of the declaration
    return module.__file__, start_line_number, end_line_number


def get_source_code_location_of_object(object_to_get_location_for: ModuleType | type[Any]) -> tuple[str | None, int | None, int | None]:
    """Retrieves the path to the source code file that contains the specified object, as well as the start and end line numbers of the object's
    definition. The object can be a module, a class, a method, a or function. If the source code file path is not available, or the object's
    definition cannot be found, then None is returned for all three values.

    Args:
        object_to_get_location_for (ModuleType | type[Any]): The object for which the source code location is to be retrieved.

    Returns:
        tuple[str | None, int | None, int | None]: Returns a tuple containing the path to the source code file that contains the specified object,
            as well as the start and end line numbers of the object's definition. If the source code file path is not available, or the object's
            definition cannot be found, then None is returned for all three values.
    """

    # Gets the absolute path to the source code file that contains the object; if the object is not a module, class, method, function, traceback,
    # frame, or code object, but, for example, a built-in function, a built-in class, a built-in module, or a property, then a TypeError is raised
    # because built-in objects are directly implemented in C and no source code is available, and properties can presumable not be located as they
    # consist of two methods that may be defined in separate code files when one of them is defined in a base class; in both cases, None is returned
    # for the source file path, start line number, and end line number
    try:
        source_file_path = inspect.getsourcefile(object_to_get_location_for)
    except TypeError:
        source_file_path = None
    if source_file_path is None:
        return None, None, None

    # Retrieves the line number at which the object's definition starts and the line number where it ends; if the source code file cannot be
    # retrieved, then an OSError is raised, if the object is not a module, class, method, function, traceback, frame, or code object, but, for
    # example, a built-in function, a built-in class, a built-in module, or a property, then a TypeError is raised because built-in objects are
    # directly implemented in C and no source code is available, and properties can presumable not be located as they consist of two methods that may
    # not come directly one after another; in both cases, None is returned for the source file path, start line number, and end line number
    try:
        source, start_line_number = inspect.getsourcelines(object_to_get_location_for)
        end_line_number = start_line_number + len(source) - 1
        return source_file_path, start_line_number, end_line_number
    except (OSError, TypeError):
        return None, None, None


def linkcode_resolve(domain: LanguageDomain, info: dict[str, str]) -> str | None:
    """Determines the URL to the source code corresponding to the object in the specified domain with the provided information. This function was
    adapted from https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73.

    Args:
        domain (LanguageDomain): Specified the language domain the object is in. Can be one of "py", "c", "cpp", or "javascript".
        info (dict[str, str]): A dictionary, which contains further information about the object for which the URL is to be retrieved. Depending on
            the domain, the following keys are guaranteed to be present:
             - domain is "py":
               - "module", which contains the name of the module.
               - "fullname", which contains the full name of the object.
             - domain is "c":
               - "names", which contains a list of names for the object.
             - domain is "cpp":
               - "names", which contains a list of names for the object.
             - domain is "javascript":
               - "object", which is the name of the object.
               - "fullname", which is the name of the item.

    Returns:
        str | None: Returns the URL of the source code corresponding to the specified object. If no URL is available, then None is returned.
    """

    # We are only interested in Python source code URLs
    if domain != 'py' or 'module' not in info or 'fullname' not in info:
        return None

    # Gets the module in which the object resides for which the source code URL is to be retrieved
    module: ModuleType | None = sys.modules.get(info['module'])
    if module is None:
        return None

    # Gets the object for which the source code URL is to be retrieved
    object_to_get_url_for: ModuleType | type[Any] | None = get_object_by_name(info['fullname'], module)
    if object_to_get_url_for is None:
        return None

    # Unfortunately, type aliases for Literal[...]'s (such as LanguageDomain) are not supported by the getsourcefile function, because they are not
    # real types, but objects of the typing._LiteralGenericAlias class, and getsourcefile only supports modules, classes, methods, and functions;
    # therefore, the only option we have is that we find the declaration of the type alias in the file of the module it is located in and then use
    # the line number of the type alias declaration to generate the URL to the source code file on GitHub; otherwise, we can use the getsourcefile and
    # getsourcelines functions to generate the URL to the source code file on GitHub
    if is_type_alias_for_literal(object_to_get_url_for):
        object_file_path, start_line_number, end_line_number = get_source_code_location_of_literal_type_alias(info['fullname'], module)
    else:
        object_file_path, start_line_number, end_line_number = get_source_code_location_of_object(object_to_get_url_for)

    # If the source code file path, the start line number, or the end line number is not available, then we cannot generate a URL to the source code
    if object_file_path is None or start_line_number is None or end_line_number is None:
        return None

    # The path to the source code file is absolute, so it must be converted to a path relative to the top module's path (example: source code file
    # path: /a/b/c/corelay/d/e.py, top module path: /a/b/c/corelay, and what we need is corelay/d/e.py); the path returned by the __file__ attribute
    # of the top module points to the module's __init__.py file, e.g., /a/b/c/corelay/__init__.py, so we need to remove the __init__.py part to get
    # the top module's path; then, we also need to remove the directory that the top module is contained in, e.g., /a/b/c; this is needed because the
    # function that turns the absolute into a relative path removes the common prefix of the two paths, which would mean that we would end up with
    # d/e.py instead of corelay/d/e.py
    top_module = get_top_level_module(info['module'])
    if top_module is None or top_module.__file__ is None:
        return None
    top_module_path = os.path.abspath(os.path.join(os.path.dirname(top_module.__file__), '..'))
    object_file_path = os.path.relpath(object_file_path, top_module_path)

    # Composes the URL to the source code file on GitHub and returns it
    return f'https://github.com/virelay/corelay/blob/{LATEST_GIT_TAG}/source/{object_file_path}#L{start_line_number}-L{end_line_number}'


def setup(app: Sphinx) -> None:
    """Is invoked by Sphinx, when this build configuration file is executed.

    Args:
        app (Sphinx): The Sphinx application.
    """

    # Sets the name of the directory into which generated documentation files are to be written and makes sure that the directory exists (this is done
    # by signing up for the config-inited event, which is emitted when the configuration has been fully initialized)
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect(
        'config-inited',
        lambda app, _: os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)
    )


# Sets the basic project information
project = 'CoRelAy'
project_copyright = f'{datetime.now().year}, CoRelAy'
author = 'CoRelAy Contributors'

# Specifies the paths for the directories that contain extra templates and static files, and configures the favicon for the HTML pages
templates_path = ['_templates']
html_static_path = ['_static']
html_favicon = '_static/favicon.svg'

# Specifies a list of patterns, relative to source directory, that match files and directories to ignore when looking for source files, in this case,
# nothing needs to be excluded
exclude_patterns: Sequence[str] = []

# Specifies the Sphinx extensions that are used by this documentation
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.datatemplates',
    'sphinx_copybutton',
    'sphinx_rtd_theme'
]

# Registers the pybtex.style.formatting plugin, which is used to format citations of bibliography entries to the format
# "[<first-author> et al., <year>]"
register_plugin('pybtex.style.formatting', 'author_year_style', AuthorYearStyle)

# Configures the Sphinx plugin, which adds a copy button to code blocks
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = '\\'
copybutton_here_doc_delimiter = 'EOT'

# Configures the theme of the HTML pages, which is set to the Read the Docs theme that was installed as a plugin
html_theme = 'sphinx_rtd_theme'

# Configures the Sphinx plugin, which allows us to insert external links to the GitHub repository using ":repo:`file-path`" instead of the full URL
LATEST_GIT_TAG = get_latest_git_tag()
extlinks = {
    'repo': (
        f'https://github.com/virelay/corelay/blob/{LATEST_GIT_TAG}/%s',
        '%s'
    )
}

# Configures the Sphinx plugin, which allows BibTeX citations to be inserted into documentation
bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'author_year_style'
bibtex_reference_style = 'author_year'
