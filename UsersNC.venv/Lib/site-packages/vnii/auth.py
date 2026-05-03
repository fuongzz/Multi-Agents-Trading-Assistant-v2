'\nAPI Key Authentication for vnii\n\nSimple, focused authentication using Vnstock API keys only.\nGitHub authentication has been removed.\n\nDesign:\n  - Single authentication method: API Key\n  - Device registration via vnstock API\n  - Unified device tracking with vnai and installer\n'
_O='auth_method'
_N='device_limit'
_M='devices_used'
_L='status'
_K='unlimited'
_J='is_registered'
_I='hw_info.json'
_H='api_key.json'
_G='https://vnstocks.com'
_F='deviceLimit'
_E='devicesUsed'
_D=False
_C='tier'
_B='api_key'
_A='device_id'
import json,logging,os,requests
from pathlib import Path
from typing import Dict,Optional
log=logging.getLogger(__name__)
class APIKeyAuthStrategy:
	'\n    API Key-based authentication (only supported method)\n\n    Uses Vnstock system API key for device registration, license verification,\n    and package installation.\n    '
	def __init__(A,project_dir,api_base_url=_G):B=project_dir;A.project_dir=B;A.api_base_url=api_base_url.rstrip('/');A.api_key_path=B/_H;A.session=requests.Session();A.name='API Key'
	def get_credentials(A):
		'Get API key from file or environment';C=os.getenv('VNSTOCK_API_KEY')
		if C:B=f"[{A.name}] Using API key from environment variable";log.debug(B);return C
		if A.api_key_path.exists():
			try:
				with open(A.api_key_path,'r')as D:
					E=json.load(D);C=E.get(_B)
					if C and C.strip():B=f"[{A.name}] Using API key from api_key.json";log.debug(B);return C.strip()
			except(json.JSONDecodeError,ValueError)as F:B=f"[{A.name}] api_key.json is corrupted: {F}";log.warning(B)
		else:B=f"[{A.name}] api_key.json not found at: {A.api_key_path}";log.debug(B)
	def verify_credentials(B,credential):
		'Verify API key and license status.\n        \n        Smart registration logic:\n        - First time: Register device with full info (IDE detection, etc)\n        - Subsequent times: Only verify license (no re-registration)\n        - If device_id already exists: API returns success without device limit error\n        ';E=credential
		try:
			A=f"[{B.name}] Getting device information...";log.debug(A)
			try:from.device import get_device_info_for_api as F;C=F();A=f"[{B.name}] Using unified device module";log.debug(A)
			except ImportError as D:A=f"[{B.name}] device module not available: {D}\nPlease ensure vnii is properly installed.";log.error(A);raise RuntimeError(A)
			G=B._is_device_registered(C.get(_A))
			if G:A=f"[{B.name}] Device already registered, verifying license only...";log.debug(A);return B._verify_license_only(E,C)
			A=f"[{B.name}] First time registration, registering device...";log.debug(A);H=B._register_device(E,C);B._save_registration_state(C.get(_A));return H
		except requests.exceptions.RequestException as D:A='Network error during license verification: {}\nPlease check your connection and try again.'.format(D);raise SystemExit(A)
	def _is_device_registered(C,device_id):
		'Check if device is already registered (from hw_info.json)'
		try:
			A=C.project_dir/_I
			if A.exists():
				with open(A,'r')as D:B=json.load(D);return B.get(_A)==device_id and B.get(_J,_D)
		except Exception as E:log.debug(f"Could not check registration state: {E}")
		return _D
	def _save_registration_state(D,device_id):
		'Save device registration state to hw_info.json'
		try:
			A=D.project_dir/_I;B={}
			if A.exists():
				with open(A,'r')as C:B=json.load(C)
			B[_A]=device_id;B[_J]=True;B['registered_at']=str(Path.ctime(A))if A.exists()else None
			with open(A,'w')as C:json.dump(B,C,indent=2)
			log.debug('Device registration state saved')
		except Exception as E:log.debug(f"Could not save registration state: {E}")
	def _verify_license_only(A,credential,device_info):
		'Verify license without re-registering device';C=device_info;B=credential;I={_B:B,_A:C.get(_A)};J='{}/api/vnstock/license/verify'
		try:
			F=A.session.post(J.format(A.api_base_url),json=I,timeout=30)
			if F.status_code==200:D=F.json();E=D.get(_C,'free');G=D.get(_E,0);H=D.get(_F,_K);K=f"[{A.name}] License verified. Tier: {E}, Devices: {G}/{H}";log.debug(K);L=A._get_username(B);M=f"License recognized and verified. Tier: {E}";return{_L:M,'user':L,_C:E,_M:G,_N:H,_A:C.get(_A),_O:_B}
		except Exception as N:log.debug(f"License verification failed: {N}")
		return A._register_device(B,C)
	def _register_device(G,credential,device_info):
		'Register device with full information';P='unknown';O='machine_info';N='os_version';M='os_type';L='device_name';K=credential;C=device_info;Q={_B:K,_A:C.get(_A),L:C.get(L),M:C.get(M),N:C.get(N),O:C.get(O)};R='{}/api/vnstock/auth/device-register';A=G.session.post(R.format(G.api_base_url),json=Q,timeout=30)
		if A.status_code!=200:
			D=A.json();J=D.get('error',f"HTTP {A.status_code}")
			if A.status_code==429:H=D.get(_E,P);E=D.get(_F,P);F=D.get(_C,'your');S=D.get('canReset',_D);T='\n💡 You can remove devices at: https://vnstocks.com/login'if S else'';B=f"""❌ License Error: Device limit exceeded
   Your {F} plan allows {E} device(s) per OS.
   Currently registered: {H}/{E}
   {J}{T}

Please remove unused devices or upgrade your subscription.""";raise SystemExit(B)
			elif A.status_code==403:B=f"❌ License Error: Subscription required\n   {J}\n\nPlease visit https://vnstocks.com/store to purchase a membership.";raise SystemExit(B)
			else:B=f"❌ License Error: Device registration failed\n   Status: {A.status_code}\n   Message: {J}\n\nPlease check your API key and try again.";raise SystemExit(B)
		I=A.json();F=I.get(_C,'free');H=I.get(_E,0);E=I.get(_F,_K);U=I.get('isNewRegistration',True);B=f"[{G.name}] Device registration successful. Tier: {F}, Devices: {H}/{E}";log.debug(B);V=G._get_username(K);W=f"License recognized and verified. Tier: {F}";return{_L:W,'user':V,_C:F,_M:H,_N:E,_A:C.get(_A),_O:_B,'is_new_registration':U}
	def _get_username(A,api_key):
		'Get username from API';C='Unknown'
		try:
			D={'Authorization':f"Bearer {api_key}"};E='{}/api/vnstock/user/profile';B=A.session.get(E.format(A.api_base_url),headers=D,timeout=10)
			if B.status_code==200:F=B.json();return F.get('username',C)
		except Exception as G:H=f"[{A.name}] Could not fetch username: {G}";log.debug(H)
		return C
def authenticate(project_dir,api_base_url=_G):
	'\n    Authenticate using API key.\n    \n    Args:\n        project_dir: Directory containing api_key.json\n        api_base_url: Base URL for vnstock API\n        \n    Returns:\n        License info dict on success\n        \n    Raises:\n        SystemExit on authentication failure\n    ';A=project_dir;B=APIKeyAuthStrategy(A,api_base_url);C=B.get_credentials()
	if not C:
		from.utils import is_google_colab as F;D=A/_H;E=f"""❌ License Error: No API key found.
   Expected location: {D}

Please either:
1. Set VNSTOCK_API_KEY environment variable, or
2. Create api_key.json file at: {D}
"""
		if F():E+='\n💡 For Google Colab:\n   Your api_key.json should be in Google Drive:\n   /content/drive/MyDrive/.vnstock/api_key.json\n   \n   Make sure Google Drive is mounted first!'
		raise SystemExit(E)
	return B.verify_credentials(C)