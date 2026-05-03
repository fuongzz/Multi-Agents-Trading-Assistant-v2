_A=None
from typing import List,Any,Dict
import pandas as pd
def validate_xy(x:List[Any]=_A,y:List[Any]=_A)->_A:
	if x is _A or y is _A:return
	if len(x)!=len(y):raise ValueError(f"Length mismatch: len(x)={len(x)} vs len(y)={len(y)}")
def validate_data(df:pd.DataFrame,required_columns:List[str])->_A:
	A=[A for A in required_columns if A not in df.columns]
	if A:B=', '.join(A);raise ValueError(f"Missing required columns: {B}")