"""
Vnstock API Client
Handles authentication, license verification, and package downloads
"""

import json
import logging
import os
import platform
import requests
import sys
import tempfile
from datetime import datetime
from typing import Dict, Optional, Tuple

from . import config
from . import __version__


logger = logging.getLogger(__name__)


class VnstockAPIClient:
    """Client for Vnstock API operations"""
    
    def __init__(self, api_key: str, python_executable: str = None):
        self.api_key = api_key
        self.base_url = config.base_url.rstrip('/')
        self.python_executable = python_executable or sys.executable
        
        # Use unified device ID from vnai (single source of truth)
        self.device_id = None
        self._load_device_id()
        
        # Setup requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                f'VnstockInstaller/{__version__} '
                f'({platform.system()} {platform.release()})'
            )
        })
        
        logger.info("VnstockAPIClient initialized")
        logger.debug(f"Device ID: {self.device_id}")

    def _load_device_id(self):
        """Load device ID from vnai"""
        try:
            from vnai.scope.profile import inspector
            self.device_id = inspector.fingerprint()
            logger.info(f"Using vnai device ID: {self.device_id}")
        except ImportError as e:
            logger.error(f"vnai not available: {e}")
            raise ImportError(
                "vnai is required for device identification. "
                "Please install it with: pip install vnai"
            ) from e

    def refresh_device_id(self):
        """Reload device ID (call after installing vnai)"""
        # Clear module cache to ensure fresh import
        for key in list(sys.modules.keys()):
            if key.startswith('vnai'):
                del sys.modules[key]
        
        self._load_device_id()

    def register_device(self) -> Tuple[bool, str, Dict]:
        """Register current device with server"""
        try:
            logger.info("Registering device with server...")
            
            # Collect system info for registration
            system_info = {
                'platform': platform.platform(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                )
            }
            
            payload = {
                'api_key': self.api_key,
                'device_id': self.device_id,
                'device_name': platform.node(),
                'os_type': platform.system().lower(),
                'os_version': platform.release(),
                'machine_info': system_info
            }
            
            response = self.session.post(
                f'{self.base_url}/api/vnstock/auth/device-register',
                json=payload,
                timeout=config.API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()

                # Save and verify API key
                key_saved = self.save_api_key()
                if not key_saved:
                    logger.warning(
                        "Device registered but API key save failed"
                    )

                # Create user info file
                self.create_user_info()

                tier = data.get('tier', 'unknown')
                logger.info(f"Device registered: tier={tier}")
                return True, "Device registered successfully", data
            
            elif response.status_code == 429:
                data = response.json()
                error_msg = data.get('message', 'Device limit exceeded')
                logger.error(f"Registration failed: {error_msg}")
                return False, error_msg, {}
            
            else:
                error_data = response.json()
                error_msg = error_data.get(
                    'error',
                    f'HTTP {response.status_code}'
                )
                logger.error(f"Registration failed: {error_msg}")
                return False, error_msg, {}
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"Registration error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, {}
    
    def list_available_packages(self) -> Tuple[bool, any]:
        """Get list of available packages"""
        try:
            logger.debug("Fetching available packages...")
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = self.session.get(
                f'{self.base_url}/api/vnstock/packages/list',
                headers=headers,
                timeout=config.API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    packages = data.get('data', {}).get('accessible', [])
                    logger.info(f"Found {len(packages)} accessible packages")
                    return True, {
                        'packages': [
                            {
                                'name': pkg['name'],
                                'displayName': pkg.get(
                                    'displayName', pkg['name']
                                ),
                                'description': pkg.get('description', ''),
                                'version': pkg.get('version', '3.0.0'),
                                'available': True
                            }
                            for pkg in packages
                        ]
                    }
                return True, data
            else:
                error_data = response.json()
                error_msg = error_data.get(
                    'error', f'HTTP {response.status_code}'
                )
                logger.error(f"Failed to list packages: {error_msg}")
                return False, error_msg
        
        except Exception as e:
            error_msg = f"Error fetching packages: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def download_package(
        self,
        package_name: str,
        install_dir: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Download package tarball"""
        try:
            logger.info(f"Starting download for {package_name}")
            
            # Get download URL
            payload = {
                'device_id': self.device_id,
                'package_name': package_name
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = self.session.post(
                f'{self.base_url}/api/vnstock/packages/download',
                json=payload,
                headers=headers,
                timeout=config.API_TIMEOUT
            )
            
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get(
                    'error',
                    f'HTTP {response.status_code}'
                )
                logger.error(f"Failed to get download URL: {error_msg}")
                return False, error_msg
            
            download_info = response.json()
            download_url = download_info['downloadUrl']
            
            # Download file
            logger.info(f"Downloading {package_name}...")
            download_response = self.session.get(
                download_url, timeout=300
            )
            
            if download_response.status_code != 200:
                msg = f"Download failed: HTTP {download_response.status_code}"
                logger.error(msg)
                return False, msg
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                suffix='.tar.gz', delete=False
            ) as temp_file:
                temp_file.write(download_response.content)
                temp_tar_path = temp_file.name
            
            logger.debug(f"Saved to temp file: {temp_tar_path}")
            return True, temp_tar_path
        
        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def save_installation_info(self, additional_data: dict = None):
        """Save installation info to user_install.json"""
        try:
            # Get IP address
            try:
                ip_resp = requests.get(
                    "https://api.ipify.org?format=json",
                    timeout=5
                )
                ip_address = ip_resp.json().get("ip", "unknown")
            except Exception:
                ip_address = "unknown"
            
            install_info = {
                "installation_time": datetime.now().isoformat(),
                "device_id": self.device_id,
                "os": platform.system(),
                "os_version": platform.version(),
                "ip_address": ip_address,
                "python_version": (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
            }
            
            if additional_data:
                install_info.update(additional_data)
            
            with open(config.USER_INFO_FILE, "w") as f:
                json.dump(install_info, f, indent=2)
            
            logger.info(
                f"Installation info saved to {config.USER_INFO_FILE}"
            )
        
        except Exception as e:
            logger.warning(f"Could not save installation info: {e}")
    
    def save_api_key(self) -> bool:
        """Save API key to config directory with detailed tracking"""
        try:
            config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            # Check if old key exists
            old_key = None
            if config.API_KEY_FILE.exists():
                try:
                    with open(config.API_KEY_FILE, "r") as f:
                        old_data = json.load(f)
                        old_key = old_data.get("api_key")
                except Exception as e:
                    logger.warning(f"Could not read old key: {e}")

            # Determine if updating or creating new
            is_update = old_key is not None
            is_changed = old_key != self.api_key

            # Save new key
            api_key_data = {"api_key": self.api_key}
            with open(config.API_KEY_FILE, "w") as f:
                json.dump(api_key_data, f, indent=2)

            # Log appropriately
            if is_update:
                if is_changed:
                    logger.info(
                        f"✓ API key UPDATED at {config.API_KEY_FILE}"
                    )
                    logger.debug(
                        f"  Old: {old_key[:10] if old_key else '?'}..."
                    )
                    logger.debug(f"  New: {self.api_key[:10]}...")
                else:
                    logger.info(
                        "✓ API key confirmed (unchanged) "
                        f"at {config.API_KEY_FILE}"
                    )
            else:
                logger.info(
                    f"✓ API key CREATED at {config.API_KEY_FILE}"
                )

            return True

        except Exception as e:
            logger.error(f"✗ Failed to save API key: {e}", exc_info=True)
            return False

    def update_api_key(self, new_api_key: str) -> Tuple[bool, str]:
        """
        Update API key in memory and file
        (for user switching keys without re-registration)

        Returns: (success, message)
        """
        try:
            if not new_api_key or not new_api_key.strip():
                return False, "API key cannot be empty"

            old_key = self.api_key
            self.api_key = new_api_key.strip()

            logger.info("API key updated in memory")
            logger.debug(f"  Old: {old_key[:10]}...")
            logger.debug(f"  New: {self.api_key[:10]}...")

            return True, "API key updated successfully"

        except Exception as e:
            logger.error(f"Failed to update API key: {e}")
            return False, f"Update failed: {str(e)}"

    def create_user_info(
        self, user: str = None, email: str = None
    ) -> bool:
        """
        Create user.json file with installation info
        (similar to setup_wizard.py for compatibility)
        """
        try:
            config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            
            # Get IP address
            try:
                ip_resp = requests.get(
                    "https://api.ipify.org?format=json",
                    timeout=5
                )
                ip_address = ip_resp.json().get("ip", "unknown")
            except Exception:
                ip_address = "unknown"
            
            # Python info
            python_info = {
                'version': (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                'executable': sys.executable,
                'is_virtual_env': self._is_virtual_environment(),
                'virtual_env_path': (
                    os.environ.get("VIRTUAL_ENV", None)
                )
            }
            
            # User info
            user_info = {
                "user": user or "vnstock_installer",
                "email": email or "unknown",
                "uuid": self.device_id,  # Use unified device ID from vnai
                "os": platform.system(),
                "os_version": platform.version(),
                "ip": ip_address,
                "cwd": os.getcwd(),
                "python": python_info,
                "time": datetime.now().isoformat(),
                "device_id": self.device_id  # Use unified device ID from vnai
            }
            
            # Save user.json
            user_info_file = config.CONFIG_DIR / "user.json"
            with open(user_info_file, "w") as f:
                json.dump(user_info, f, indent=2)
            
            logger.info(f"User info saved to {user_info_file}")
            return True
        
        except Exception as e:
            logger.warning(f"Could not create user info: {e}")
            return False
    
    def _is_virtual_environment(self) -> bool:
        """Check if running in virtual environment"""
        return (
            hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and
             sys.base_prefix != sys.prefix)
        )
