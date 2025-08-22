# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../geti_sdk"))

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
with open(f"{root_path}/geti_sdk/_version.py", encoding="utf-8") as init_file:
    for line in init_file:
        line = line.strip()
        if line.startswith("__version__"):
            VERSION = line.split("=")[1].strip().strip('"')


# -- Project information -----------------------------------------------------

project = "Intel® Geti™ SDK"
copyright = "2024 Intel Corporation"  # noqa: A001
author = "Ludo Cornelissen"

# The full version, including alpha/beta/rc tags
release = f"v{VERSION}"

master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosectionlabel", "myst_parser"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "exclude_members": "__weakref__",
}

# -- Options for inclusion of markdown files --------------------------------
# myst_gfm_only = True
myst_heading_anchors = 4

# source_suffix = [".rst"]

# -- Options for jupyter notebook conversion --------------------------------
nbsphinx_execute = "never"
