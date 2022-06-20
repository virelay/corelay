import sys
import os
from subprocess import run, CalledProcessError
import inspect
import pkg_resources

from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.plugin import register_plugin


# -- Project information -----------------------------------------------------
project = 'corelay'
copyright = '2021, corelay'
author = 'corelay'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinxcontrib.datatemplates',
    'sphinxcontrib.bibtex',
]


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if isinstance(obj, property):
        return True
    return None


def setup(app):
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect('config-inited', config_inited_handler)
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)


templates_path = ['_templates']
exclude_patterns = []

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"

html_theme = 'sphinx_rtd_theme'
html_favicon = '_static/favicon.svg'
html_static_path = ['_static']

bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'author_year_style'
bibtex_reference_style = 'author_year'


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield f'[{entry.persons["author"][0].last_names[0]} et al., {entry.fields["year"]}]'


class AuthorYearStyle(PlainStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin('pybtex.style.formatting', 'author_year_style', AuthorYearStyle)


def getrev():
    try:
        revision = run(
            ['git', 'describe', '--tags', 'HEAD'],
            capture_output=True,
            check=True,
            text=True
        ).stdout[:-1]
    except CalledProcessError:
        revision = 'master'

    return revision


REVISION = getrev()

extlinks = {
    'repo': (
        f'https://github.com/virelay/corelay/blob/{REVISION}/%s',
        '%s'
    )
}

LINKCODE_URL = (
    f'https://github.com/virelay/corelay/blob/{REVISION}'
    '/src/{filepath}#L{linestart}-L{linestop}'
)


# revised from https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return LINKCODE_URL.format(filepath=filepath, linestart=linestart, linestop=linestop)
