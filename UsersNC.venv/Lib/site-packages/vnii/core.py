'\nCore Orchestrator for vnii\n\nSimplified to focus on API Key authentication.\nGitHub OAuth and all legacy systems removed.\n'
import logging
from pathlib import Path
from.analytics import Analytics,system_info,packages_installed
from.auth import APIKeyAuthStrategy,authenticate
log=logging.getLogger(__name__)
class VnstockInitializer:
	'\n    Main initializer for vnstock licensed packages.\n    \n    SIMPLIFIED for API Key authentication only.\n    Device tracking unified with vnai and installer.\n    '
	def __init__(A,target,analytics_class=Analytics):'\n        Initialize vnstock licensing system.\n        \n        Args:\n            target: Target package name\n            analytics_class: Analytics provider (default: Analytics wrapper)\n        ';B=True;from.utils import get_vnstock_directory as C;A.project_dir=C();A.project_dir.mkdir(parents=B,exist_ok=B);A.home_dir=A.project_dir.parent;A.id_dir=A.project_dir/'id';A.id_dir.mkdir(parents=B,exist_ok=B);A.target=target;A.analytics_class=analytics_class;A._init_managers()
	def _init_managers(A):'Initialize managers (simplified to essentials only)';A.analytics=A.analytics_class(A.project_dir,A.id_dir,A.target,None,None);A.auth_strategy=APIKeyAuthStrategy(A.project_dir)
	def system_info(A):'Get system information';return system_info()
	def packages_installed(A):'Get list of installed packages';return packages_installed()
	def authenticate(A):'\n        Authenticate using API key.\n        \n        Returns:\n            License info dict on success\n            \n        Raises:\n            SystemExit on authentication failure\n        ';return authenticate(A.project_dir)