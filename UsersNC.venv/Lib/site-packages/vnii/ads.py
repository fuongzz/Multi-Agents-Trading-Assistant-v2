'\nAdvertisement Module for vnstock - API Key Auth Version\n\nQuản lý hiển thị quảng cáo dựa trên:\n- User tier (free vs paid) từ API Key authentication\n- Tần suất hiển thị theo category\n- Lịch sử tracking với vnai\n\nVersion: 0.2.0 - API Key authentication\n'
_F='verified'
_E='last_trigger'
_D='tier'
_C='ads'
_B=False
_A=None
import re,sys,json,logging
from pathlib import Path
from datetime import datetime,timedelta,timezone
from typing import Optional
log=logging.getLogger(__name__)
class AdCategory:'Phân loại quảng cáo theo mức độ quan trọng';FREE=0;MANDATORY=1;ANNOUNCEMENT=2;REFERRAL=3;FEATURE=4;GUIDE=5;SURVEY=6;PROMOTION=7;SECURITY=8;MAINTENANCE=9;WARNING=10
class AdDefinition:
	'Định nghĩa quảng cáo với tần suất'
	def __init__(A,name,freq_seconds,category):'\n        Args:\n            name: Tên quảng cáo (unique identifier)\n            freq_seconds: Tần suất hiển thị (giây)\n            category: Loại quảng cáo (AdCategory)\n        \n        Example:\n            >>> ad = AdDefinition("premium_upgrade", 604800, AdCategory.FREE)\n            >>> ad.freq.days\n            7\n        ';A.name=name;A.freq=timedelta(seconds=freq_seconds);A.category=category
class AdHistoryStore:
	'\n    Lưu trữ lịch sử hiển thị quảng cáo\n    \n    Sử dụng vnai để lấy project_dir thay vì colab_helper\n    '
	def __init__(A,project_dir=_A):
		'\n        Args:\n            project_dir: Thư mục lưu trữ (mặc định từ vnai)\n        ';B=project_dir
		if B is _A:B=A._get_vnstock_data_dir()
		A.log_file=B/'ads_log.json'
		if not A.log_file.exists():A.log_file.parent.mkdir(parents=True,exist_ok=True);A._save({_C:{}})
		log.debug(f"AdHistoryStore initialized: {A.log_file}")
	def _get_vnstock_data_dir(C):
		'Lấy thư mục data từ vnstock ggcolab hoặc fallback'
		try:from.utils import get_vnstock_directory as A;return A()
		except Exception as B:log.warning(f"Failed to get vnstock directory: {B}");return Path.home()/'.vnstock'
	def _load(A):
		'Load ads log từ file'
		try:
			with open(A.log_file,'r')as B:return json.load(B)
		except Exception as C:log.error(f"Failed to load ads log: {C}");return{_C:{}}
	def _save(A,data):
		'Save ads log to file'
		try:
			with open(A.log_file,'w')as B:json.dump(data,B,indent=2)
		except Exception as C:log.error(f"Failed to save ads log: {C}")
	def get_last(C,name):
		'\n        Lấy lần hiển thị cuối cùng của quảng cáo\n        \n        Args:\n            name: Tên quảng cáo\n        \n        Returns:\n            datetime của lần hiển thị cuối, hoặc None\n        ';D=C._load().get(_C,{});B=D.get(name,{}).get(_E)
		if B:
			try:
				A=datetime.fromisoformat(B)
				if A.tzinfo is _A:A=A.replace(tzinfo=timezone.utc)
				return A
			except Exception as E:log.error(f"Failed to parse timestamp: {E}");return
	def log_show(B,name,timestamp=_A):
		'\n        Ghi nhận quảng cáo đã hiển thị\n        \n        Args:\n            name: Tên quảng cáo\n            timestamp: Thời điểm hiển thị (mặc định: now)\n        ';E='total_count';A=timestamp
		if A is _A:A=datetime.now(timezone.utc)
		C=B._load();D=C.setdefault(_C,{}).setdefault(name,{E:0});D[E]+=1;D[_E]=A.isoformat();B._save(C);log.debug(f"Logged ad show: {name} at {A}")
	def log_skip(A,name,reason):'\n        Ghi nhận quảng cáo bị bỏ qua\n        \n        Args:\n            name: Tên quảng cáo\n            reason: Lý do bỏ qua (sponsor, frequency, etc.)\n        ';log.debug(f"Skipped ad: {name} - Reason: {reason}")
class AdScheduler:
	'\n    Lập lịch và kiểm soát hiển thị quảng cáo\n    \n    Sử dụng auth_strategy để xác định user tier (free/paid)\n    '
	def __init__(A,project_dir=_A):'\n        Args:\n            project_dir: Thư mục lưu trữ history\n        ';A.history=AdHistoryStore(project_dir=project_dir);A._license_info=_A;log.debug('AdScheduler initialized')
	def _get_license_info(A):
		'\n        Lấy thông tin license từ auth_strategy\n        \n        Returns:\n            dict với keys: tier, user, verified\n        '
		if A._license_info is not _A:return A._license_info
		try:from.core import VnstockInitializer as B;C=B(target='vnstock');A._license_info=C.authenticate();D=A._license_info.get(_D);log.debug(f"License info retrieved: tier={D}");return A._license_info
		except Exception as E:log.error(f"Failed to get license info: {E}");return{_D:'free','user':'unknown',_F:_B}
	def is_paid_user(C):'\n        Kiểm tra user có phải paid tier không\n        \n        Returns:\n            True nếu user là premium/enterprise, False nếu free\n        ';D=C._get_license_info();A=D.get(_D,'free').lower();E=['premium','enterprise','professional','sponsor'];B=A in E;log.debug(f"User tier: {A}, is_paid: {B}");return B
	def should_show(B,ad):
		'\n        Kiểm tra có nên hiển thị quảng cáo không\n        \n        Logic:\n        1. FREE ads: Chỉ hiện cho free users\n        2. MANDATORY ads: Hiện cho tất cả\n        3. Frequency check: Tuân theo tần suất định nghĩa\n        4. Colab check: Chỉ hiện nếu authenticated trong Colab\n        \n        Args:\n            ad: AdDefinition object\n        \n        Returns:\n            True nếu nên hiển thị, False nếu skip\n        \n        Example:\n            >>> ad = AdDefinition("upgrade", 604800, AdCategory.FREE)\n            >>> scheduler = AdScheduler()\n            >>> should_show = scheduler.should_show(ad)\n        ';A=ad
		if A.category==AdCategory.FREE:
			if B.is_paid_user():B.history.log_skip(A.name,'paid_user');return _B
		C=B.history.get_last(A.name)
		if C:
			D=datetime.now(timezone.utc)-C
			if D<A.freq:B.history.log_skip(A.name,'frequency');log.debug(f"Ad {A.name} skipped: shown {D.total_seconds():.0f}s ago, needs {A.freq.total_seconds():.0f}s");return _B
		if'google.colab'in sys.modules:
			E=B._get_license_info()
			if not E.get(_F,_B):B.history.log_skip(A.name,'colab_not_authenticated');log.debug('Ad skipped: Not authenticated in Colab');return _B
		log.debug(f"Ad {A.name} should be shown");return True
def parse_meta_frequency(html):
	'\n    Parse <meta name="ad-freq" content="3/wk"> thành số giây\n    \n    Hỗ trợ:\n    - /d (ngày): 86400 giây\n    - /wk (tuần): 604800 giây\n    - /m (tháng): 2592000 giây (30 ngày)\n    \n    Args:\n        html: HTML string chứa meta tag\n    \n    Returns:\n        Số giây tương ứng với tần suất, 0 nếu không tìm thấy\n    \n    Example:\n        >>> html = \'<meta name="ad-freq" content="1/wk">\'\n        >>> parse_meta_frequency(html)\n        604800.0\n    ';C=.0;F='<meta[^>]*name=["\\\']ad-freq["\\\'][^>]*content=["\\\']([^"\\\']+)["\\\']';D=re.search(F,html)
	if not D:return C
	G=D.group(1).strip().lower()
	try:
		E=re.match('(\\d+(?:\\.\\d+)?)/(\\w+)',G)
		if not E:return C
		H,A=E.groups();B=float(H)
		if A=='d':return B*86400
		elif A=='wk':return B*604800
		elif A=='m':return B*2592000
		else:log.warning(f"Unknown frequency unit: {A}");return B
	except Exception as I:log.error(f"Failed to parse frequency: {I}");return C
def create_ad_from_meta(html,name,category=AdCategory.FREE):
	'\n    Tạo AdDefinition từ HTML meta tag\n    \n    Args:\n        html: HTML string chứa meta tag\n        name: Tên quảng cáo\n        category: AdCategory (mặc định: FREE)\n    \n    Returns:\n        AdDefinition object\n    \n    Example:\n        >>> html = \'<meta name="ad-freq" content="1/wk">\'\n        >>> ad = create_ad_from_meta(html, "upgrade_banner", AdCategory.FREE)\n        >>> ad.freq.days\n        7\n    ';A=parse_meta_frequency(html)
	if A==0:A=604800
	return AdDefinition(name,A,category)