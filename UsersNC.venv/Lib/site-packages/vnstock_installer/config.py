"""
Configuration module for Vnstock Installer
Manages global settings and constants
"""

import os
import sys
from enum import IntEnum
from pathlib import Path


class UIOpenMode(IntEnum):
    """UI opening mode enumeration"""
    NONE = 0
    CHROME_OR_EDGE = 1
    DEFAULT_BROWSER = 2


# Frontend assets directory
FRONTEND_ASSET_FOLDER = Path(__file__).parent / "web"

# API Configuration
DEFAULT_BASE_URL = "https://vnstocks.com"
API_TIMEOUT = 30  # seconds

# Python Configuration
SUPPORTED_PYTHON_VERSIONS = ['3.10', '3.11', '3.12', '3.13', '3.14']
DEFAULT_RECURSION_LIMIT = sys.getrecursionlimit()

# Package Index Configuration
# Vnstocks index for vnii (licensing) and other packages
# Can be updated if vnstocks changes their index URL
VNSTOCKS_INDEX_URL = 'https://vnstocks.com/api/simple'

# Package Dependencies
# Core dependencies (pure Python, no compilation required)
# Vnii (licensing) is installed from vnstocks.com extra index
# Versions are now fetched dynamically from REQUIREMENTS_URL
REQUIREMENTS_URL = "https://vnstocks.com/files/requirements.txt"

REQUIRED_DEPENDENCIES = [
    'vnai',  # Device fingerprinting (must be first for unified device ID)
    'vnii',  # Licensing module from vnstocks index
    'numpy',
    'pandas',
    'requests',
    'beautifulsoup4',
    'aiohttp',
    'nest-asyncio',
    'pydantic',
    'psutil',
    'pyarrow',
    'openpyxl',
    'tqdm',
    'panel',
    'pyecharts',
    'pta-reload',
]

# Optional dependencies that require build tools (C++, CMake)
# These will be installed with --prefer-binary to avoid compilation
OPTIONAL_COMPILED_DEPENDENCIES = [
    'duckdb',
    'vnstock_ezchart',  # Contains wordcloud
]

# Vnstock Branding Colors
COLORS = {
    'primary': '#4CAF50',      # Energetic Green
    'accent': '#8C52FF',       # Dreamy Purple
    'background': '#FAFAFA',   # Soft White/Gray
    'card': '#FFFFFF',         # Pure White
    'text': '#1A1A1A',         # Dark Gray/Black
    'border': '#E0E0E0',       # Light Gray
    'error': '#E74C3C',        # Red
    'warning': '#F39C12',      # Orange
    'success': '#27AE60',      # Green
}

# User Configuration Paths
HOME_DIR = Path.home()
CONFIG_DIR = HOME_DIR / ".vnstock"
API_KEY_FILE = CONFIG_DIR / "api_key.json"
SETTINGS_FILE = CONFIG_DIR / "gui_settings.json"
USER_INFO_FILE = CONFIG_DIR / "user_install.json"
LOG_FILE = CONFIG_DIR / "vnstock_installer.log"

# Ensure config directory exists
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Runtime Configuration (modified by command-line arguments)
ui_open_mode = UIOpenMode.CHROME_OR_EDGE
base_url = os.environ.get('VNSTOCK_BASE_URL', DEFAULT_BASE_URL)
language_hint = None
supplied_ui_configuration = None
default_output_directory = os.path.abspath("output")

# Installation state
temporary_directory = None
installation_running = False
selected_python = None
use_venv = True
venv_path = HOME_DIR / ".venv"
