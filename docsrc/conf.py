import sphinx_rtd_theme
import caldera
import datetime

# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.doctest',
#     'sphinx.ext.intersphinx',
#     'sphinx.ext.mathjax',
#     'sphinx.ext.napoleon',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.githubpages',
# ]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx_rtd_theme"
]

author = 'Justin D Vrana '
project = 'caldera'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

# autosummary_generate = True

# styling
templates_path = ['_templates']
# html_logo = '_static/img/pyg_logo_text.svg'
html_static_path = ['_static']
html_context = {'css_files': ['_static/css/custom.css']}
html_style = 'css/theme.css'
source_suffix = '.rst'
master_doc = 'index'

version = caldera.__version__
release = caldera.__version__

html_theme = 'sphinx_rtd_theme'

# intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
    'navigation_depth': 2,
}

# add_module_names = False
def setup(app):
    app.add_stylesheet("css/theme.css")
# def setup(app):
#     def skip(app, what, name, obj, skip, options):
#         members = [
#             '__init__',
#             '__repr__',
#             '__weakref__',
#             '__dict__',
#             '__module__',
#         ]
#         return True if name in members else skip
#
#     app.connect('autodoc-skip-member', skip)