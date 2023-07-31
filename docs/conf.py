# lsqfitgp/docs/conf.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
#
# This file is part of lsqfitgp.
#
# lsqfitgp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# lsqfitgp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'lsqfitgp'
author = 'Giacomo Petrillo'

from datetime import datetime
now = datetime.now()
year = '2020'
if now.year > int(year):
    year += '-' + str(now.year)
copyright = year + ', ' + author


# # The full version, including alpha/beta/rc tags
import lsqfitgp
release = lsqfitgp.__version__
version = release
project += ' ' + version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx.ext.napoleon', # alternative to numpydoc, broken last time I tried
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_title = f'{project} documentation'

html_theme_options = dict(
    description = 'A general purpose Gaussian process regression module',
    fixed_sidebar = True,
    github_button = True,
    github_type = 'star',
    github_repo = 'lsqfitgp',
    github_user = 'Gattocrucco',
    show_relbars = True,
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'


# -- Other options -------------------------------------------------

autoclass_content = 'both'
autodoc_preserve_defaults = True

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

default_role = 'py:obj'

intersphinx_mapping = dict(
    gvar=('https://gvar.readthedocs.io/en/latest', None),
    lsqfit=('https://lsqfit.readthedocs.io/en/latest', None),
    scipy=('https://docs.scipy.org/doc/scipy/reference', None),
    numpy=('https://numpy.org/doc/stable', None),
)
