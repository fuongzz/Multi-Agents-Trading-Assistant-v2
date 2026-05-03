"""
Installation logic for Vnstock packages
Handles package extraction, dependency installation, and verification
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config
from .api import VnstockAPIClient


logger = logging.getLogger(__name__)


# Bilingual messages (Vietnamese / English)
MESSAGES = {
    'installing_deps': 'Đang cài đặt gói phụ thuộc / Installing dependencies',
    'installing_core': 'Đang cài đặt gói cốt lõi / Installing core dependencies',
    'installing_optional': 'Đang cài đặt gói tùy chọn / Installing optional dependencies',
    'deps_ready': 'Gói phụ thuộc đã sẵn sàng / Dependencies ready',
    'deps_prepared': 'Đã chuẩn bị gói phụ thuộc / Dependencies prepared',
    'deps_error': 'Lỗi gói phụ thuộc / Dependencies error',
    'deps_installed': 'Đã cài đặt gói phụ thuộc / Dependencies installed',
    'deps_timeout': 'Hết thời gian cài đặt (tiếp tục) / Installation timeout (continuing)',
    'downloading': 'Đang tải xuống / Downloading',
    'extracting': 'Đang giải nén / Extracting',
    'installing': 'Đang cài đặt / Installing',
    'verifying': 'Đang xác minh / Verifying',
    'completed': 'Hoàn thành / Completed',
    'failed': 'Thất bại / Failed',
}


def get_message(key: str, package_name: str = None) -> str:
    """Get bilingual message"""
    msg = MESSAGES.get(key, key)
    if package_name:
        msg = f"{msg} {package_name}"
    return msg



class VnstockInstaller:
    """Handles installation of Vnstock packages"""
    
    def __init__(
        self,
        api_client: Optional[VnstockAPIClient],
        python_executable: str,
        use_venv: bool = False,
        venv_path: Optional[str] = None
    ):
        self.api_client = api_client
        self.python_executable = python_executable
        self.uv_executable = 'uv'  # Default to system 'uv'
        self.use_venv = use_venv
        self.venv_path = venv_path
        self.venv_python = None
        self.installed_packages = []
        self.failed_packages = []
        self.skipped_dependencies = []  # Track skipped optional deps
        
        logger.info(f"VnstockInstaller initialized with Python: {python_executable}")
        logger.info(f"Virtual environment: use_venv={use_venv}, path={venv_path}")
        
        # Setup virtual environment if requested
        if use_venv and venv_path:
            self._setup_venv()
    
    def _ensure_uv_installed(self) -> bool:
        """Ensure uv is installed, installing via pip if necessary"""
        # Try to resolve uv first
        if self._resolve_uv_executable():
            logger.info(f"uv is available at {self.uv_executable}")
            return True

        logger.info("uv not found, installing via pip...")
        try:
            subprocess.run(
                [self.python_executable, '-m', 'pip', 'install', 'uv'],
                capture_output=True,
                check=True,
                timeout=60,
                **self._get_subprocess_kwargs()
            )
            logger.info("uv installed successfully")
            
            # Resolve again after install
            if self._resolve_uv_executable():
                logger.info(f"uv resolved at {self.uv_executable}")
                return True
                
            logger.warning("uv installed but not found in PATH. Attempting to guess location...")
            return self._guess_uv_location()
            
        except Exception as e:
            logger.error(f"Failed to install uv: {e}")
            return False

    def _resolve_uv_executable(self) -> bool:
        """Check if current uv_executable works or find it in PATH"""
        # Check current
        try:
            subprocess.run(
                [self.uv_executable, '--version'],
                capture_output=True,
                check=True,
                **self._get_subprocess_kwargs()
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Check PATH
        path = shutil.which('uv')
        if path:
            self.uv_executable = path
            return True
            
        return False
        
    def _guess_uv_location(self) -> bool:
        """Guess uv location based on python executable"""
        # Usually in same dir as python
        bin_dir = os.path.dirname(self.python_executable)
        uv_name = 'uv.exe' if platform.system() == 'Windows' else 'uv'
        uv_path = os.path.join(bin_dir, uv_name)
        
        if os.path.exists(uv_path):
            self.uv_executable = uv_path
            return True
            
        # Try user site bin as last resort (Linux/Mac)
        # We can't easily guess exact user site bin without shelling out, 
        # but typically ~/.local/bin
        # Let's rely on what we found.
        return False

    def _setup_venv(self):
        """Create and setup virtual environment using uv"""
        try:
            expanded_path = os.path.expanduser(self.venv_path)
            # We don't need to makedirs explicitly as uv will handle it, 
            # but getting the path absolute is good.
            
            # Check if venv already exists
            venv_config = os.path.join(expanded_path, 'pyvenv.cfg')
            if not os.path.exists(venv_config):
                logger.info(f"Creating virtual environment at {expanded_path}")
                
                # Ensure uv is installed first
                if not self._ensure_uv_installed():
                    raise RuntimeError("Could not install uv to bootstrap virtual environment")

                # Use uv to create venv
                result = subprocess.run(
                    [self.uv_executable, 'venv', expanded_path, '--python', self.python_executable],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    **self._get_subprocess_kwargs()
                )
                
                if result.returncode != 0:
                    msg = f"Failed to create venv with uv: {result.stderr}"
                    logger.error(msg)
                    raise RuntimeError(msg)
            else:
                logger.info(f"Virtual environment already exists at {expanded_path}")
            
            # Update python executable to use venv python
            if platform.system() == 'Windows':
                venv_python = os.path.join(expanded_path, 'Scripts', 'python.exe')
            else:
                venv_python = os.path.join(expanded_path, 'bin', 'python')
            
            if os.path.exists(venv_python):
                self.venv_python = venv_python
                self.python_executable = venv_python
                logger.info(f"Virtual environment setup complete: {venv_python}")
            else:
                msg = f"Failed to find python in venv at {venv_python}"
                logger.error(msg)
                raise RuntimeError(msg)
        
        except Exception as e:
            msg = f"Failed to setup virtual environment: {str(e)}"
            logger.error(msg, exc_info=True)
            raise

    def _validate_package_spec(self, pkg_spec: str) -> Tuple[str, str]:
        """
        Parse package specification to get import name and version constraint
        Returns: (import_name, version_constraint)
        """
        # Split by common version operators
        for op in ['>=', '==', '<=', '>', '<', '!=', '~=']:
            if op in pkg_spec:
                pkg_name, version = pkg_spec.split(op, 1)
                break
        else:
            pkg_name = pkg_spec
            version = None

        # Convert package name to import name
        # Handle special cases
        import_name = pkg_name.strip()
        special_imports = {
            'beautifulsoup4': 'bs4',
            'pyyaml': 'yaml',
            'pillow': 'PIL',
            'pydantic_core': 'pydantic_core',
            'nest-asyncio': 'nest_asyncio',
            'pta-reload': 'pta_reload',
            'importlib-metadata': 'importlib_metadata',
            'pyarrow': 'pyarrow',
            'tqdm': 'tqdm',
            'panel': 'panel',
            'pyecharts': 'pyecharts',
            'duckdb': 'duckdb',
            'vnstock_ezchart': 'vnstock_ezchart',
        }

        import_name = special_imports.get(import_name, import_name)
        import_name = import_name.replace('-', '_')
        return import_name, version

    def _check_single_package(
        self, pkg_spec: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a single package is already installed
        Returns: (is_installed, error_reason)
        """
        import_name, _ = self._validate_package_spec(pkg_spec)

        try:
            # Try to import the package
            result = subprocess.run(
                [
                    self.python_executable, '-c',
                    f'import {import_name}; print("OK")'
                ],
                capture_output=True,
                text=True,
                timeout=10,
                **self._get_subprocess_kwargs()
            )

            if result.returncode == 0 and 'OK' in result.stdout:
                logger.debug(
                    f"✓ Package '{import_name}' already installed"
                )
                return True, None

            # If import fails, check pip list as fallback
            # We can use uv pip list now or just stick to python -m pip as standard
            parts = pkg_spec.split('>=')[0].split('==')[0]
            pkg_name_normalized = parts.split('<=')[0]
            
            # Using standard pip module check is safer for compatibility
            # Using uv pip show is safer and works in pip-less venvs
            result = subprocess.run(
                [
                    self.uv_executable, 'pip', 'show', pkg_name_normalized,
                    '--python', self.python_executable
                ],
                capture_output=True,
                text=True,
                timeout=10,
                **self._get_subprocess_kwargs()
            )

            if result.returncode == 0:
                logger.debug(
                    f"✓ Package '{pkg_name_normalized}' found in pip"
                )
                return True, None
            else:
                return False, "Not found in pip list"

        except subprocess.TimeoutExpired:
            return False, "Package check timed out"
        except Exception as e:
            return False, str(e)

    def _install_packages_uv(
        self,
        packages: List[str],
        prefer_binary: bool = False,
        upgrade: bool = False,
        timeout: int = 300
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Install a batch of packages using uv
        Returns: (successful_packages, failed_packages_with_reasons)
        """
        if not packages:
            return [], []

        successful = []
        failed = []

        try:
            # Build uv pip install command
            # We use 'uv pip install' targeting the current environment
            cmd = [self.uv_executable, 'pip', 'install']
            
            if upgrade:
                cmd.append('--upgrade')
            
            # Use remote requirements.txt as constraint file
            # This ensures we get the exact versions defined remotely
            # while only installing the specific packages we request here.
            cmd.extend(['-c', config.REQUIREMENTS_URL])
            
            # If we are targeting a specific python env (venv), uv pip install handles 
            # VIRTUAL_ENV env var or we can specify --python if needed explicitly,
            # but generally setting VIRTUAL_ENV is best practice or just activating it.
            # However, since we are calling it as a subprocess, we can point it to the python executable
            # using the --python flag.
            cmd.extend(['--python', self.python_executable])
            
            # Add extra index for vnstocks
            cmd.extend([
                '--extra-index-url',
                config.VNSTOCKS_INDEX_URL
            ])
            
            # uv handles binary preference automatically/better, but we can't strict enforce like pip --prefer-binary
            # In uv, we don't usually need it for speed, but if compilation fails, maybe.
            # For now, let's omit prefer-binary flag as uv optimizes resolution.

            cmd.extend(packages)

            logger.info(f"Installing {len(packages)} packages using uv...")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Set environment for uv to detect venv if needed
            env = os.environ.copy()
            if self.venv_path:
                 env['VIRTUAL_ENV'] = os.path.abspath(self.venv_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                **self._get_subprocess_kwargs()
            )

            if result.returncode == 0:
                # Verify each package was actually installed
                for pkg_spec in packages:
                    is_installed, error = self._check_single_package(pkg_spec)
                    # Normalize name for report
                    pkg_name_part = pkg_spec
                    if is_installed:
                        successful.append(pkg_name_part)
                    else:
                        err_msg = error or "Verification failed"
                        failed.append((pkg_name_part, err_msg))
            else:
                # All packages failed or uv error
                logger.error(f"uv install failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr[:500]}")

                for pkg_spec in packages:
                    failed.append((pkg_spec, "uv install failed"))

        except subprocess.TimeoutExpired:
            logger.error(f"Installation timeout after {timeout}s")
            for pkg_spec in packages:
                failed.append((pkg_spec, "Installation timeout"))
        except Exception as e:
            logger.error(f"Batch installation error: {str(e)}", exc_info=True)
            for pkg_spec in packages:
                failed.append((pkg_spec, str(e)))

        return successful, failed

    def install_dependencies(
        self,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """Install required dependencies using uv with remote version constraints"""
        try:
            logger.info("=" * 60)
            logger.info("DEPENDENCY INSTALLATION PHASE (UV)")
            logger.info("=" * 60)

            if progress_callback:
                progress_callback(0.05, get_message('installing_deps'))

            # Ensure uv is installed
            if not self._ensure_uv_installed():
                return False, "Could not install uv package manager"

            # Phase 1: Analyze core dependencies
            # We now rely on uv to resolve versions using the remote requirements.txt as a constraint file.
            # This means we pass package names directly, and uv will ensure the correct versions are installed.
            logger.info("\n[Phase 1] Analyzing core dependencies...")
            
            logger.info(f"  Refining dependency list: {len(config.REQUIRED_DEPENDENCIES)} packages")


            # Phase 2: Install core dependencies
            logger.info(f"\n[Phase 2] Installing/Verifying core packages...")

            if progress_callback:
                msg = get_message('installing_core')
                progress_callback(0.25, msg)

            # Split vnai from other dependencies for priority installation
            vnai_pkg = 'vnai'
            other_pkgs = [p for p in config.REQUIRED_DEPENDENCIES if p != vnai_pkg]
            
            # Phase 2a: Install vnai first (Priority & Latest)
            logger.info("  [Phase 2a] Installing vnai (Priority)...")
            vnai_successful, vnai_failed = self._install_packages_uv(
                [vnai_pkg],
                upgrade=True,  # Always upgrade vnai
                timeout=120
            )
            
            if vnai_successful:
                logger.info("  ✓ vnai installed successfully. Refreshing Device ID...")
                try:
                    # Refresh device ID now that vnai is available
                    if hasattr(self.api_client, 'refresh_device_id'):
                        self.api_client.refresh_device_id()
                except Exception as e:
                    logger.warning(f"Could not refresh device ID: {e}")
            else:
                logger.error("  ✗ Failed to install vnai (Critical)")
                # We continue but warn, or should we stop?
                # For now let's verify later.

            # Phase 2b: Install remaining core dependencies
            logger.info(f"\n[Phase 2b] Installing remaining core packages ({len(other_pkgs)})...")

            core_successful, core_failed = \
                self._install_packages_uv(
                    other_pkgs,
                    prefer_binary=False,
                    timeout=300
                )
            
            # Merge results for reporting
            core_successful.extend(vnai_successful)
            if vnai_failed:
                core_failed.extend(vnai_failed)

            logger.info("\n[Phase 2 Results]")
            logger.info(f"  Successful (checked/installed): {len(core_successful)}")
            if core_failed:
                logger.error(f"  Failed: {len(core_failed)}")
                for pkg_name, reason in core_failed:
                    logger.error(f"    ✗ {pkg_name}: {reason}")

            # Phase 3: Verify core dependencies
            logger.info("\n[Phase 3] Verifying core dependencies...")
            verification_failed = []
            
            # Simple check if they exist now
            for pkg_name in config.REQUIRED_DEPENDENCIES:
                 is_installed, error = self._check_single_package(pkg_name)
                 if not is_installed:
                     verification_failed.append((pkg_name, error))
                     logger.error(f"  ✗ {pkg_name}: {error}")
                 else:
                     logger.info(f"  ✓ {pkg_name}")

            if verification_failed:
                num_failed = len(verification_failed)
                error_msg = f"Critical: {num_failed} dependencies failed"
                logger.error(f"\n{error_msg}")
                return False, error_msg

            logger.info("\nAll core dependencies verified")

            # Phase 4: Optional dependencies
            logger.info("\n[Phase 4] Optional dependencies...")
            
            # For optional, we might want to be more careful?
            # uv check is fast.
            optional_to_install = config.OPTIONAL_COMPILED_DEPENDENCIES

            num_opt_install = len(optional_to_install)
            logger.info(f"  {num_opt_install} optional to install")

            optional_successful = []
            optional_failed = []

            if optional_to_install:
                if progress_callback:
                    progress_callback(0.6, get_message('installing_optional'))

                optional_successful, optional_failed = self._install_packages_uv(
                    optional_to_install,
                    prefer_binary=True,
                    timeout=120
                )

            logger.info("\n[Phase 4 Results]")
            if optional_successful:
                logger.info(f"  ✓ Successful: {len(optional_successful)}")
            if optional_failed:
                 logger.warning(f"  ⚠ Failed: {len(optional_failed)}")
                 for pkg_name, reason in optional_failed:
                     self.skipped_dependencies.append(pkg_name)

            if progress_callback:
                progress_callback(1.0, get_message('deps_ready'))

            # Summary
            msg = get_message('deps_installed')
            if optional_failed:
                 msg += f" (skipped {len(optional_failed)})"
            
            return True, msg

        except subprocess.TimeoutExpired:
            msg = get_message('deps_timeout')
            logger.warning(msg)
            if progress_callback:
                progress_callback(1.0, get_message('deps_prepared'))
            return True, msg
        except Exception as e:
            msg = f"{get_message('deps_error')}: {str(e)}"
            logger.error(msg, exc_info=True)
            if progress_callback:
                progress_callback(1.0, get_message('deps_error'))
            return False, msg
    
    def bootstrap_vnai(self) -> Tuple[bool, str]:
        """
        Bootstrap vnai installation without API client
        Used by UI to ensure vnai exists before registration
        """
        try:
            logger.info("Bootstrapping vnai...")
            if not self._ensure_uv_installed():
                return False, "Could not install uv"
                
            successful, failed = self._install_packages_uv(
                ['vnai'],
                upgrade=True,
                timeout=180
            )
            
            if successful:
                return True, "vnai bootstrapped successfully"
            else:
                reason = failed[0][1] if failed else "Unknown error"
                return False, f"Failed to bootstrap vnai: {reason}"
                
        except Exception as e:
            logger.error(f"Bootstrap error: {e}", exc_info=True)
            return False, str(e)
            
    def install_package(
        self,
        package_name: str,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """Install a single Vnstock package"""
        try:
            logger.info(f"Installing {package_name}...")
            
            if progress_callback:
                msg = get_message('downloading', package_name)
                progress_callback(0.2, msg)
            
            # Download package
            success, result = self.api_client.download_package(package_name)
            if not success:
                logger.error(f"Download failed: {result}")
                self.failed_packages.append((package_name, result))
                return False, result
            
            temp_tar_path = result
            
            if progress_callback:
                progress_callback(0.5, get_message('extracting', package_name))
            
            try:
                # Extract package
                install_path = config.temporary_directory or tempfile.mkdtemp()
                extract_dir = os.path.join(install_path, package_name)
                os.makedirs(extract_dir, exist_ok=True)
                
                logger.debug(f"Extracting to {extract_dir}")
                
                with tarfile.open(temp_tar_path, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
                
                # Find setup directory
                extracted_files = os.listdir(extract_dir)
                setup_dir = None
                
                for item in extracted_files:
                    item_path = os.path.join(extract_dir, item)
                    if os.path.isdir(item_path):
                        setup_py = os.path.join(item_path, 'setup.py')
                        pyproject = os.path.join(item_path, 'pyproject.toml')
                        if (os.path.exists(setup_py) or
                                os.path.exists(pyproject)):
                            setup_dir = item_path
                            break
                
                if not setup_dir:
                    msg = "No setup.py or pyproject.toml found"
                    logger.error(msg)
                    self.failed_packages.append((package_name, msg))
                    return False, msg
                
                if progress_callback:
                    msg = get_message('installing', package_name)
                    progress_callback(0.8, msg)
                
                # Install package with no dependencies to avoid
                # build tools issues (dependencies handled separately)
                # Dependencies will be handled separately
                logger.info(f"Installing from {setup_dir}")
                result = subprocess.run(
                    [
                        self.uv_executable, 'pip', 'install', '--no-deps', '-q',
                        '--python', self.python_executable,
                        setup_dir
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    **self._get_subprocess_kwargs()
                )
                
                # Verify installation
                if result.returncode == 0:
                    if self._verify_package_import(package_name):
                        logger.info(f"{package_name} installed successfully")
                        self.installed_packages.append(package_name)
                        if progress_callback:
                            progress_callback(
                                1.0,
                                get_message('completed', package_name)
                            )
                        return True, get_message('completed')
                    else:
                        msg = "Package cannot be imported"
                        logger.warning(f"{package_name}: {msg}")
                        self.failed_packages.append((package_name, msg))
                        return False, msg
                else:
                    # Check if error is due to missing build tools
                    stderr = result.stderr.lower()
                    if 'building wheel' in stderr or 'cmake' in stderr:
                        msg = "Requires build tools (Visual C++ or CMake)"
                        logger.warning(f"{package_name}: {msg}")
                        msg = ("Install Visual Studio Build Tools "
                               "for full support")
                        logger.warning(msg)
                    else:
                        msg = "Installation failed (pip error)"
                        logger.error(f"{package_name}: {msg}")
                    
                    logger.error(result.stderr[:500])
                    self.failed_packages.append((package_name, msg))
                    return False, msg
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_tar_path):
                    os.unlink(temp_tar_path)
        
        except Exception as e:
            msg = f"Installation error: {str(e)}"
            logger.error(f"{package_name}: {msg}", exc_info=True)
            self.failed_packages.append((package_name, msg))
            return False, msg
    
    def _verify_package_import(self, package_name: str) -> bool:
        """Verify package can be imported"""
        try:
            # Try different import patterns for Vnstock packages
            import_names = []
            
            # Pattern 1: vnstock_ta -> vnstock.ta
            if package_name.startswith('vnstock_'):
                module_name = package_name.replace('vnstock_', 'vnstock.')
                import_names.append(module_name)
            
            # Pattern 2: Original name
            import_names.append(package_name)
            
            # Pattern 3: Without underscore
            import_names.append(package_name.replace('_', ''))
            
            for import_name in import_names:
                result = subprocess.run(
                    [
                        self.python_executable,
                        '-c',
                        f'import {import_name}; print("OK")'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    **self._get_subprocess_kwargs()
                )
                
                if result.returncode == 0 and 'OK' in result.stdout:
                    logger.info(
                        f"Package {package_name} verified "
                        f"as {import_name}"
                    )
                    return True
            
            # If all patterns fail, check if package is in pip list
            result = subprocess.run(
                [
                    self.uv_executable, 'pip', 'list', '--format=freeze',
                    '--python', self.python_executable
                ],
                capture_output=True,
                text=True,
                timeout=30,
                **self._get_subprocess_kwargs()
            )
            
            if result.returncode == 0:
                installed = result.stdout.lower()
                pkg_normalized = package_name.replace('_', '-').lower()
                if (package_name.lower() in installed or
                        pkg_normalized in installed):
                    logger.info(f"Package {package_name} found in pip list")
                    return True
            
            logger.warning(f"Could not verify import for {package_name}")
            return False
            
        except Exception as e:
            logger.warning(
                f"Import verification error for {package_name}: {e}"
            )
            # Check pip list as fallback
            try:
                result = subprocess.run(
                    [
                        self.uv_executable, 'pip', 'show', package_name,
                        '--python', self.python_executable
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    **self._get_subprocess_kwargs()
                )
                if result.returncode == 0:
                    logger.info(f"Package {package_name} found via pip show")
                    return True
            except Exception:
                pass
            return False
    
    def _get_subprocess_kwargs(self):
        """Get kwargs for subprocess to hide console on Windows"""
        kwargs = {}
        if platform.system() == 'Windows':
            # CREATE_NO_WINDOW = 0x08000000
            kwargs['creationflags'] = 0x08000000
        return kwargs
    
    def get_installation_summary(self) -> Dict:
        """Get summary of installation results"""
        return {
            'installed': self.installed_packages,
            'failed': self.failed_packages,
            'skipped_deps': self.skipped_dependencies,
            'total': len(self.installed_packages) + len(self.failed_packages),
            'success_count': len(self.installed_packages),
            'fail_count': len(self.failed_packages)
        }
