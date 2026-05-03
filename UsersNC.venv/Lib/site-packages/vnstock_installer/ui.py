"""
UI Module - Eel-based web interface for Vnstock Installer
Handles frontend-backend communication and UI state management
"""

import eel
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess

import sys
import importlib 
from eel import chrome, edge

from . import config
from .api import VnstockAPIClient
from .installer import VnstockInstaller
from . import validation
from . import __version__


logger = logging.getLogger(__name__)


# Initialize Eel
eel.init(str(config.FRONTEND_ASSET_FOLDER))


# Global state
current_api_client = None
current_installer = None
installation_progress = 0


def get_device_id() -> str:
    """Generate unique device ID based on hostname and Python executable"""
    unique_str = f"{platform.node()}-{sys.executable}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:32]


@eel.expose
def initialise():
    """Called by UI when opened"""
    logger.info("UI initializing...")
    
    # Check for critical dependencies
    system_health = {
        'vnai': False,
        'requests': False,
        'warnings': []
    }
    
    try:
        import vnai
        system_health['vnai'] = True
    except ImportError:
        system_health['warnings'].append(
            "Gói 'vnai' chưa được cài đặt - Sẽ tự động tải và cài đặt khi Đăng ký thiết bị"
        )
        
    try:
        import requests
        system_health['requests'] = True
    except ImportError:
        system_health['warnings'].append(
            "Gói 'requests' chưa được cài đặt (Missing 'requests')"
        )

    return {
        'version': __version__,
        'suppliedConfig': config.supplied_ui_configuration,
        'languageHint': config.language_hint,
        'supportedPythonVersions': config.SUPPORTED_PYTHON_VERSIONS,
        'defaultVenvPath': str(config.venv_path),
        'baseUrl': config.base_url,
        'systemHealth': system_health # Return health status to UI
    }


@eel.expose
def detect_python_versions():
    """Detect available Python versions"""
    logger.info("Detecting Python versions...")
    
    versions = []
    seen = set()
    
    # Current Python
    try:
        result = subprocess.run(
            [sys.executable, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_str = result.stdout.strip() or result.stderr.strip()
            parts = version_str.split()
            if len(parts) >= 2:
                ver = parts[1]
                major, minor = map(int, ver.split('.')[:2])
                if major == 3 and minor >= 10:
                    # Check if version meets minimum requirement (3.11+)
                    meets_requirement = minor >= 11
                    warning = (
                        "" if meets_requirement
                        else " ⚠️ vnstock-pipeline yêu cầu Python 3.11+"
                    )
                    
                    versions.append({
                        'display': (
                            f"Python {ver} ({sys.executable}){warning}"
                        ),
                        'executable': sys.executable,
                        'version': f"{major}.{minor}",
                        'meetsRequirement': meets_requirement
                    })
                    seen.add(sys.executable)
    except Exception as e:
        logger.warning(f"Failed to detect current Python: {e}")
    
    # System Pythons
    candidates = [
        'python3', 'python', 'python3.13', 'python3.12',
        'python3.11', 'python3.10'
    ]
    
    for exe in candidates:
        try:
            full_path = shutil.which(exe)
            if not full_path or full_path in seen:
                continue
            
            result = subprocess.run(
                [full_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_str = result.stdout.strip() or result.stderr.strip()
                parts = version_str.split()
                if len(parts) >= 2:
                    ver = parts[1]
                    major, minor = map(int, ver.split('.')[:2])
                    if major == 3 and minor >= 10:
                        meets_requirement = minor >= 11
                        warning = (
                            "" if meets_requirement
                            else " ⚠️ vnstock-pipeline yêu cầu 3.11+"
                        )
                        versions.append({
                            'display': f"{exe} ({ver}) - {full_path}{warning}",
                            'executable': full_path,
                            'version': f"{major}.{minor}",
                            'meetsRequirement': meets_requirement
                        })
                        seen.add(full_path)
        except Exception:
            continue
    
    logger.info(f"Found {len(versions)} Python versions")
    return versions


@eel.expose
def validate_api_key(api_key):
    """Validate API key"""
    is_valid, message = validation.validate_api_key(api_key)
    return {'valid': is_valid, 'message': message}


@eel.expose
def register_device(api_key, python_exe):
    """Register device with Vnstock API"""
    global current_api_client, current_installer
    
    try:
        logger.info("Registering device...")
        
        # Check Python version and warn
        result = subprocess.run(
            [python_exe, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_str = result.stdout.strip() or result.stderr.strip()
            parts = version_str.split()
            if len(parts) >= 2:
                ver = parts[1]
                major, minor = map(int, ver.split('.')[:2])
                if major == 3 and minor < 11:
                    warning_msg = (
                        f"⚠️ Python {ver} không hỗ trợ vnstock-pipeline "
                        f"(yêu cầu 3.11+)"
                    )
                    logger.warning(warning_msg)
        
        # Check if vnai is available, if not bootstrap it
        try:
            import vnai
        except ImportError:
            logger.info("vnai not found, attempting to bootstrap...")
            # Use temporary installer without API client
            temp_installer = VnstockInstaller(None, python_exe)
            success, msg = temp_installer.bootstrap_vnai()
            
            if success:
                logger.info("Bootstrap successful, reloading imports")
                importlib.invalidate_caches()
                try:
                    import vnai
                    logger.info("vnai imported successfully after bootstrap")
                except ImportError as e:
                    logger.error(f"Failed to import vnai after bootstrap: {e}")
                    return {
                        'success': False,
                        'message': f"Lỗi khởi tạo môi trường (vnai import): {e}"
                    }
            else:
                logger.error(f"Bootstrap failed: {msg}")
                return {
                    'success': False,
                    'message': f"Lỗi cài đặt vnai (Bootstrap): {msg}"
                }

        # Set global API client AND installer
        current_api_client = VnstockAPIClient(api_key, python_exe)
        current_installer = VnstockInstaller(current_api_client, python_exe)
        success, message, data = current_api_client.register_device()
        
        return {
            'success': success,
            'message': message,
            'tier': data.get('tier', 'unknown'),
            'deviceLimit': data.get('deviceLimit', 'unknown'),
            'devicesUsed': data.get('devicesUsed', 0)
        }
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return {
            'success': False,
            'message': str(e)
        }


@eel.expose
def list_packages():
    """List available packages"""
    global current_api_client

    if not current_api_client:
        msg = 'API client not initialized'
        return {'success': False, 'message': msg}

    try:
        success, data = current_api_client.list_available_packages()
        return {
            'success': success,
            'packages': data.get('packages', []) if success else [],
            'message': str(data) if not success else ''
        }
    except Exception as e:
        logger.error(f"List packages error: {e}", exc_info=True)
        return {'success': False, 'message': str(e)}


@eel.expose
def update_api_key(new_api_key: str) -> dict:
    """Update API key without re-registration"""
    global current_api_client

    if not current_api_client:
        return {
            'success': False,
            'message': 'API client not initialized'
        }

    try:
        success, message = current_api_client.update_api_key(
            new_api_key
        )

        if success:
            # Also save to file
            saved = current_api_client.save_api_key()
            if saved:
                logger.info("API key updated and saved")
                return {'success': True, 'message': message}
            else:
                return {
                    'success': False,
                    'message': 'Updated but failed to save'
                }
        else:
            return {'success': False, 'message': message}

    except Exception as e:
        logger.error(f"Update API key error: {e}", exc_info=True)
        return {'success': False, 'message': str(e)}


@eel.expose
def start_installation(
    python_exe, selected_packages, use_venv, venv_path
):
    """Start package installation"""
    global current_api_client, current_installer, installation_progress
    
    if not current_api_client:
        return {
            'success': False,
            'message': 'Vui lòng đăng nhập và đăng ký thiết bị trước'
        }
    
    try:
        logger.info(
            f"Starting installation of {len(selected_packages)} packages"
        )
        
        # Sort packages in correct installation order
        package_order = [
            'vnstock_data',
            'vnstock_ta',
            'vnstock_news',
            'vnstock_pipeline'
        ]
        
        def get_install_priority(pkg_name):
            """Get package install priority (lower = install first)"""
            for i, pattern in enumerate(package_order):
                if pattern in pkg_name:
                    return i
            return 999  # Unknown packages go last
        
        # Sort selected packages by priority
        sorted_packages = sorted(
            selected_packages,
            key=get_install_priority
        )
        logger.info(f"Installation order: {sorted_packages}")
        
        # Create installer with venv settings
        if use_venv:
            venv_path_final = venv_path or str(config.venv_path)
            current_installer = VnstockInstaller(
                current_api_client,
                python_exe,
                use_venv=True,
                venv_path=venv_path_final
            )
        else:
            current_installer = VnstockInstaller(
                current_api_client,
                python_exe
            )
        
        # Install dependencies
        def dep_progress(progress, message):
            global installation_progress
            installation_progress = progress * 0.3
            eel.updateProgress(installation_progress, message)
        
        success, msg = current_installer.install_dependencies(dep_progress)
        if not success:
            return {'success': False, 'message': msg}
        
        # Install packages in correct order
        total = len(sorted_packages)
        for i, pkg in enumerate(sorted_packages):
            def pkg_progress(progress, message):
                global installation_progress
                base = 0.3 + (i / total) * 0.7
                installation_progress = (
                    base + (progress / total) * 0.7
                )
                eel.updateProgress(installation_progress, message)
            
            success, msg = current_installer.install_package(
                pkg,
                pkg_progress
            )
            logger.info(
                f"{pkg}: {'success' if success else 'failed'} - {msg}"
            )
        
        # Get summary
        summary = current_installer.get_installation_summary()
        
        # Save installation info
        current_api_client.save_installation_info({
            'installed_packages': summary['installed'],
            'failed_packages': summary['failed']
        })
        
        return {
            'success': True,
            'summary': summary
        }
    except Exception as e:
        logger.error(f"Installation error: {e}", exc_info=True)
        return {'success': False, 'message': str(e)}


@eel.expose
def open_config_folder():
    """Open config folder in file explorer"""
    try:
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        if platform.system() == 'Darwin':  # macOS
            os.system(f'open "{config.CONFIG_DIR}"')
        elif platform.system() == 'Windows':
            os.system(f'explorer "{config.CONFIG_DIR}"')
        else:  # Linux
            os.system(f'xdg-open "{config.CONFIG_DIR}"')
        
        return True
    except Exception as e:
        logger.error(f"Failed to open config folder: {e}")
        return False


@eel.expose
def get_username():
    """Get current username"""
    try:
        return os.getlogin()
    except Exception:
        # Fallback to environment variable
        return (
            os.environ.get('USERNAME') or
            os.environ.get('USER') or
            'User'
        )


@eel.expose
def load_saved_api_key() -> str:
    """Try to load saved API key from config with validation"""
    try:
        if config.API_KEY_FILE.exists():
            file_size = config.API_KEY_FILE.stat().st_size
            logger.debug(f"API key file found ({file_size} bytes)")

            with open(config.API_KEY_FILE, "r") as f:
                data = json.load(f)
                api_key = data.get("api_key", "").strip()

                if api_key:
                    logger.info(
                        f"✓ Loaded saved API key ({len(api_key)} chars)"
                    )
                    return api_key
                else:
                    logger.warning(
                        "API key file exists but empty"
                    )
        else:
            logger.debug("No saved API key file found")

    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON in API key file: {e}")
    except FileNotFoundError:
        logger.debug("API key file not accessible")
    except Exception as e:
        logger.error(
            f"✗ Unexpected error loading API key: {e}",
            exc_info=True
        )

    return ""


def _can_use_chrome():
    """Check if Chrome is available"""
    chrome_path = chrome.find_path()
    return chrome_path is not None and os.path.exists(chrome_path)


def _can_use_edge():
    """Check if Edge is available"""
    return edge.find_path() is not None


def start(open_mode):
    """Start the UI using Eel"""
    try:
        chrome_available = _can_use_chrome()
        edge_available = _can_use_edge()
        
        if open_mode == config.UIOpenMode.CHROME_OR_EDGE and chrome_available:
            print("Opening in Chrome...")
            eel.start("index.html", size=(1000, 700), port=0, mode="chrome")
        elif open_mode == config.UIOpenMode.CHROME_OR_EDGE and edge_available:
            print("Opening in Edge...")
            eel.start("index.html", size=(1000, 700), port=0, mode="edge")
        elif open_mode == config.UIOpenMode.DEFAULT_BROWSER or (
            open_mode == config.UIOpenMode.CHROME_OR_EDGE and
            not chrome_available and not edge_available
        ):
            print("Opening in default browser...")
            eel.start(
                "index.html", size=(1000, 700), port=0,
                mode="user default"
            )
        else:
            from . import utils
            port = utils.get_port()
            msg = f"Server starting at http://localhost:{port}/index.html"
            print(msg)
            eel.start(
                "index.html", host="localhost", port=port, mode=None,
                close_callback=lambda x, y: None
            )
    except (SystemExit, KeyboardInterrupt):
        logger.info("UI closed")
        pass
