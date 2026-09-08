"""Sphinx configuration for the dim_red field guide.

Docstrings live in ``src/dim_red`` and are pulled in live via autodoc, so
this documentation stays in sync with the code instead of duplicating it.
Narrative pages (architecture, workflows, rationale) are written by hand in
``docs/*.md``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "dim_red"
copyright = "2026, Alessandro Serafini"
author = "Alessandro Serafini"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path: list = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / napoleon -------------------------------------------------
# Every dataclass in pipeline/config.py already carries a Google-style
# "Attributes:" docstring -- napoleon renders that as a proper field list,
# and autodoc pulls field names/types/defaults straight from the class
# signature, so the parameter reference below is never hand-duplicated.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
# NOTE: "members" is deliberately *not* enabled by default. Every config
# dataclass already documents its fields in a Google-style "Attributes:"
# docstring section (rendered by napoleon); turning on ``:members:`` on top
# of that would additionally emit each field as its own ``py:attribute``,
# colliding with the docstring-derived entry ("duplicate object
# description"). Pages that reference a class just write a bare
# ``.. autoclass::`` (docstring + constructor signature is enough); the few
# classes with real extra methods (e.g. ``SoapConfig.as_kwargs``) opt back
# in with an explicit ``:members: name1, name2`` naming only those.
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output ----------------------------------------------------------
html_theme = "furo"
html_title = "dim_red field guide"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2f6f5e",
        "color-brand-content": "#2f6f5e",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5fb39c",
        "color-brand-content": "#5fb39c",
    },
}
