'\nAnalytics Wrapper for vnii\n\nSimple wrapper around vnai.inspector for system information.\nAll complex analytics logic delegated to vnai.\n'
import logging
from typing import Dict,List
log=logging.getLogger(__name__)
def system_info():
	'\n    Get system information.\n    \n    Delegates to vnai.inspector for comprehensive system data.\n    Falls back to minimal info if vnai not available.\n    \n    Returns:\n        System information dict with platform, Python version, etc.\n    '
	try:from vnai.scope.profile import inspector as C;D=C.examine();log.debug('System info retrieved from vnai.inspector');return D
	except ImportError:log.warning('vnai not available, returning minimal system info');import platform as A;return{'platform':A.system(),'python_version':A.python_version(),'machine':A.machine()}
	except Exception as B:log.error(f"Failed to get system info: {B}");return{'error':str(B)}
def packages_installed():
	'\n    Get list of installed Python packages.\n    \n    Delegates to vnai.inspector which has optimized package scanning.\n    Falls back to importlib.metadata if vnai not available.\n    \n    Returns:\n        List of installed package names\n    '
	try:
		from vnai.scope.profile import inspector as C;A=C.scan_packages();log.debug(f"Found {len(A)} packages via vnai.inspector")
		if isinstance(A,dict):return list(A.keys())
		return A
	except ImportError:
		log.warning('vnai not available, using importlib.metadata')
		try:import importlib.metadata;D=importlib.metadata.distributions();A=[A.name for A in D];return A
		except Exception as B:log.error(f"Failed to scan packages: {B}");return[]
	except Exception as B:log.error(f"Failed to get packages: {B}");return[]
class Analytics:
	'\n    Legacy Analytics class for backward compatibility.\n    \n    Simplified to just wrap vnai.inspector functionality.\n    All webhook/tracking logic removed - handled by installer now.\n    '
	def __init__(A,project_dir,id_dir,target,RH,LH):'Initialize Analytics (legacy compatibility)';from pathlib import Path as B;A.project_dir=B(project_dir);A.id_dir=B(id_dir);A.target=target;A.RH=RH;A.LH=LH;log.debug('Analytics initialized (legacy mode)')
	def system_info(A):'Get system info (delegates to module function)';return system_info()
	def packages_installed(A):'Get installed packages (delegates to module function)';return packages_installed()
	def log_analytics_data(A,license_info):'\n        Log analytics data (deprecated).\n        \n        Analytics tracking now handled by installer.\n        This method is no-op for backward compatibility.\n        ';log.debug('log_analytics_data called (no-op, handled by installer)')