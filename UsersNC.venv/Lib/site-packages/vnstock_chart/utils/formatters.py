_A=None
from datetime import datetime
from typing import Union,List
import pandas as pd
def format_large_number(value:float,decimals:int=2)->str:
	C=decimals;A=value
	if abs(A)>=1000000000:B=f"{A/1000000000:.{C}f}B"
	elif abs(A)>=1000000:B=f"{A/1000000:.{C}f}M"
	elif abs(A)>=1000:B=f"{A/1000:.{C}f}K"
	else:B=f"{A:.{C}f}"
	if'.'in B:B=B.rstrip('0').rstrip('.')
	return B
def format_time(time_series:Union[pd.Series,List[datetime]],format_str:str=_A)->List[str]:
	F='%Y-%m-%d';B=format_str;A=time_series
	if B is _A:
		if isinstance(A,pd.Series)and len(A)>=2:
			C=_A
			for D in range(1,min(10,len(A))):
				E=(A.iloc[D]-A.iloc[D-1]).total_seconds()
				if C is _A or E<C:C=E
			if C is _A or C>=86400:B=F
			elif C>=3600:B='%Y-%m-%d %H:%M'
			else:B='%Y-%m-%d %H:%M:%S'
		else:B=F
	if isinstance(A,pd.Series):return A.dt.strftime(B).tolist()
	else:return[A.strftime(B)for A in A]
def format_number(value:Union[int,float],precision:int=2,use_k:bool=True)->str:
	B=precision;A=value
	if A is _A:return'N/A'
	if not use_k:return f"{A:,.{B}f}"
	if abs(A)>=1000000000:return f"{A/1000000000:.{B}f}B"
	elif abs(A)>=1000000:return f"{A/1000000:.{B}f}M"
	elif abs(A)>=1000:return f"{A/1000:.{B}f}K"
	else:return f"{A:.{B}f}"