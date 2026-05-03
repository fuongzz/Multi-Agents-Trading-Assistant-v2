'\nUnified Device Identification for vnii\n\nIntegrates with vnai for consistent device tracking.\nVersion: 0.2.0 - Legacy device_manager removed\n\nDesign:\n  - Primary: vnai.inspector.fingerprint()\n  - Fallback: UUID-based temporary ID\n  - Logging: Debug info for device tracking\n\nIMPORTANT: All vnai imports are lazy (inside functions) to avoid\ncircular import when vnii is imported during vnai initialization.\n'
_A='device_id'
import logging,uuid,platform
from typing import Dict
log=logging.getLogger(__name__)
def get_unified_device_id():
	'\n    Get device ID using vnai.fingerprint() as single source of truth.\n    \n    vnai is required - no fallback to ensure consistency across\n    vnstock_installer, vnii, and vnai.\n    \n    Returns:\n        32-character hex string device identifier\n    \n    Raises:\n        ImportError: If vnai is not installed\n    \n    Example:\n        >>> device_id = get_unified_device_id()\n        >>> len(device_id)\n        32\n    '
	try:from vnai.scope.profile import inspector as D;C=D.fingerprint();log.debug(f"Using vnai device_id: {C}");return C
	except ImportError as A:B='vnai is required for device identification. Please install it with: pip install vnai';log.error(f"vnai not available: {A}");raise ImportError(B)from A
	except Exception as A:B=f"Failed to get vnai device_id: {A}";log.error(B);raise
def get_device_info_for_api():
	"\n    Get device information in API-compatible format.\n    \n    Returns dict with keys:\n        - device_id: Unique 32-char identifier\n        - device_name: Hostname\n        - os_type: Operating system (lowercase)\n        - os_version: OS version string\n        - machine_info: System details dict (matches TypeScript interface)\n    \n    This format matches what vnstock API expects in TypeScript.\n    \n    Example:\n        >>> info = get_device_info_for_api()\n        >>> 'device_id' in info and 'device_name' in info\n        True\n    ";O='environment';N='python_version';M='release';L='system';K='processor';J='machine';I='machine_info';H='os_version';G='os_type';F='device_name';C='platform'
	try:
		from vnai.scope.profile import inspector as P;A=P.examine();E='Unknown';D={}
		try:from vnai.scope.device import IDEDetector as Q;E,D=Q.detect_ide()
		except(ImportError,SyntaxError,IndentationError)as B:log.debug(f"IDE detection unavailable: {type(B).__name__}: {B}")
		except Exception as B:log.debug(f"IDE detection failed: {type(B).__name__}: {B}")
		R={_A:A['machine_id'],F:A.get(C,platform.node()),G:A['os_name'].lower(),H:A.get(C,platform.release()),I:{C:A.get(C,platform.platform()),J:platform.machine(),K:platform.processor(),L:platform.system(),M:platform.release(),N:A.get(N),O:A.get(O,'unknown'),'ide_name':E,'ide_detection_method':D.get('detection_method'),'ide_frontend':D.get('frontend')}};log.debug('Using vnai for device info (API-compatible)');return R
	except Exception as B:log.debug('Using fallback device info (vnai unavailable)');return{_A:get_unified_device_id(),F:platform.node(),G:platform.system().lower(),H:platform.release(),I:{C:platform.platform(),J:platform.machine(),K:platform.processor(),L:platform.system(),M:platform.release(),'error':str(B)}}
def compare_device_ids():
	"\n    Get device ID information.\n    \n    Used during debugging to validate device identification.\n    \n    Returns dict with:\n        - device_id: The unified device ID\n        - source: Which system is being used\n        - available: Whether device ID is available\n    \n    Example:\n        >>> result = compare_device_ids()\n        >>> 'device_id' in result\n        True\n    "
	try:from vnai.scope.profile import inspector as C;A=C.fingerprint();B='vnai';log.debug(f"Device ID from vnai: {A}")
	except Exception as D:log.debug(f"vnai not available: {D}");A=get_unified_device_id();B='uuid_fallback'
	return{_A:A,'source':B,'available':True}
def get_system_info():
	'\n    Get comprehensive system information.\n    \n    Delegates to vnai.inspector if available,\n    otherwise returns empty dict.\n    \n    Returns:\n        System information dict with platform, packages, environment\n    '
	try:from vnai.scope.profile import inspector as B;log.debug('Getting system info from vnai.inspector');return B.examine()
	except ImportError:log.debug('vnai not available for system info');return{}
	except Exception as A:log.error(f"Failed to get system info: {A}");return{'error':str(A)}