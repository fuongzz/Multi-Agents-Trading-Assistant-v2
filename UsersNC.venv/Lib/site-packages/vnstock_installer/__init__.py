"""
Vnstock Installer - Professional GUI installer for Vnstock packages
Web-based desktop application using Eel framework
"""

__version__ = "3.1.1"
__author__ = "Vnstock Team"
__email__ = "support@vnstocks.com"

from . import config
from . import ui
from . import installer
from . import api
from . import validation

__all__ = [
    '__version__',
    'config',
    'ui',
    'installer',
    'api',
    'validation',
]
