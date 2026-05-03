'\nCommand-line interface for vnii\n\nSimple CLI to initialize vnstock license using API Key authentication.\n'
_A='vnstock'
import sys,logging
from pathlib import Path
from.core import VnstockInitializer
from.utils import get_vnstock_directory
try:from vnai.beam.auth import get_auth_state_manager
except ImportError:from.auth_state import get_auth_state_manager
log=logging.getLogger(__name__)
def lc_init(debug=False,package_name=_A):
	"\n    Initialize and verify vnstock license using API Key.\n\n    Args:\n        debug: Enable detailed logging (default: False)\n        package_name: Name of calling package for dedup (default: 'vnstock')\n\n    Returns:\n        License info dict on success\n        \n    Raises:\n        SystemExit on authentication failure\n        \n    Example:\n        >>> license_info = lc_init(debug=True)\n        >>> print(license_info['tier'])\n        'premium'\n    ";E=package_name;D=debug
	if not log.handlers:F=logging.StreamHandler(sys.stdout);G=logging.Formatter('%(message)s');F.setFormatter(G);log.addHandler(F)
	if D:log.setLevel(logging.DEBUG);log.debug('vnii: Debug mode enabled')
	else:log.setLevel(logging.INFO)
	H=get_vnstock_directory();C=get_auth_state_manager(H);I=C.should_show_message(E);J=VnstockInitializer(target=_A)
	try:
		A=J.authenticate();K=C._state.get('authenticated',False);C.mark_authenticated(A,E)
		if I or not K:L=A.get('user','Unknown');M=A.get('tier','free');log.info(f"✅ Authentication successful: {L} ({M})")
		log.debug(f"License info: {A}");return A
	except SystemExit as B:log.error(f"❌ Authentication failed: {B}");raise
	except Exception as B:
		log.error(f"❌ Unexpected error: {B}")
		if D:import traceback as N;N.print_exc()
		raise SystemExit(str(B))