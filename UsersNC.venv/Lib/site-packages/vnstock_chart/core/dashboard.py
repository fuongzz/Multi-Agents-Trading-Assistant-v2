_E='dark'
_D='stretch_height'
_C='stretch_both'
_B=None
_A='stretch_width'
import panel as pn
from typing import Sequence,List,Optional,Union,Dict,Any
from vnstock_chart.core.base import ChartBase
from vnstock_chart.core.env import EnvDetector,Environment
pn.extension('echarts')
class Dashboard:
	def __init__(A,charts:Sequence[ChartBase],title:str=_B,description:str=_B,theme:str=_E,height:int=_B,width:int=_B):
		D=width;C=height;B=charts;A._charts=list(B);A.title=title;A.description=description;A.theme=theme
		if len(B)>0:A.height=C or B[0].height*2;A.width=D or B[0].width
		else:A.height=C or 900;A.width=D or 1200
		if A.theme==_E:pn.config.theme=_E
		else:pn.config.theme='default'
		A.env=EnvDetector.detect()
	def _as_pane(A,chart_obj,sizing_mode=_C):return pn.pane.ECharts(chart_obj,sizing_mode=sizing_mode)
	def _create_header(A):
		D='margin-bottom';C='font-size'
		if not A.title and not A.description:return
		B=[]
		if A.title:E={C:'24px','font-weight':'bold',D:'10px'};B.append(pn.pane.HTML(f"<h1 style='text-align:center;'>{A.title}</h1>"))
		if A.description:F={C:'16px',D:'20px'};B.append(pn.pane.HTML(f"<p style='text-align:center;'>{A.description}</p>"))
		return pn.Column(*B,sizing_mode=_A)
	def layout3(A):
		if len(A._charts)<3:raise ValueError(f"Layout3 requires at least 3 charts, but only {len(A._charts)} provided")
		D,E,F=A._charts[:3];B=A._create_header();C=pn.Row(pn.Column(A._as_pane(D._chart),width=2*A.width//3),pn.Column(A._as_pane(E._chart),A._as_pane(F._chart),width=A.width//3),sizing_mode=_A,height=A.height)
		if B:return pn.Column(B,C,sizing_mode=_A)
		return C
	def layout4(A):
		if len(A._charts)<4:raise ValueError(f"Layout4 requires at least 4 charts, but only {len(A._charts)} provided")
		D,E,F,G=A._charts[:4];B=A._create_header();H=pn.Row(A._as_pane(D._chart),A._as_pane(E._chart),A._as_pane(F._chart),sizing_mode=_A,height=A.height//3);I=pn.Row(A._as_pane(G._chart),sizing_mode=_A,height=2*A.height//3);C=pn.Column(H,I,sizing_mode=_C)
		if B:return pn.Column(B,C,sizing_mode=_A)
		return C
	def layout6(A):
		if len(A._charts)<7:raise ValueError(f"Layout6 requires at least 7 charts, but only {len(A._charts)} provided")
		D,E,F=A._charts[:3];G=A._charts[3];H,I,J=A._charts[4:7];B=A._create_header();K=pn.Row(A._as_pane(D._chart),A._as_pane(E._chart),A._as_pane(F._chart),sizing_mode=_A,height=A.height//3);L=pn.Row(pn.Column(A._as_pane(G._chart),sizing_mode=_D,width=2*A.width//3),pn.Column(A._as_pane(H._chart),A._as_pane(I._chart),A._as_pane(J._chart),sizing_mode=_D,width=A.width//3),sizing_mode=_A);C=pn.Column(K,L,sizing_mode=_C)
		if B:return pn.Column(B,C,sizing_mode=_A)
		return C
	def grid(A,rows:int=2,cols:int=2):
		C=cols;B=rows;E=B*C
		if len(A._charts)<E:raise ValueError(f"Grid layout with {B}x{C} requires at least {E} charts, but only {len(A._charts)} provided")
		F=A._create_header();G=[];D=0
		for J in range(B):
			H=[]
			for K in range(C):
				if D<len(A._charts):H.append(A._as_pane(A._charts[D]._chart));D+=1
			G.append(pn.Row(*H,sizing_mode=_A))
		I=pn.Column(*G,sizing_mode=_C)
		if F:return pn.Column(F,I,sizing_mode=_A)
		return I
	def custom_layout(D,layout_spec:List[Dict[str,Any]]):
		C='children';B='layout';A=D._create_header()
		def H(spec):
			K='row';F='charts';A=spec
			if F in A:
				E=A[F]
				if isinstance(E,int):E=[E]
				G=[]
				for I in E:
					if 0<=I<len(D._charts):G.append(D._as_pane(D._charts[I]._chart))
				if A.get(B)==K:return pn.Row(*G,sizing_mode=_A,**{A:C for(A,C)in A.items()if A not in[F,B]})
				else:return pn.Column(*G,sizing_mode=_D,**{A:C for(A,C)in A.items()if A not in[F,B]})
			elif C in A:
				J=[H(A)for A in A[C]]
				if A.get(B)==K:return pn.Row(*J,sizing_mode=_A,**{A:D for(A,D)in A.items()if A not in[C,B]})
				else:return pn.Column(*J,sizing_mode=_D,**{A:D for(A,D)in A.items()if A not in[C,B]})
		E=H({C:layout_spec,B:'column'})
		if A:return pn.Column(A,E,sizing_mode=_A)
		return E
	def render(A,layout_type:str='auto'):
		G='grid';B=layout_type
		if B=='auto':
			if len(A._charts)<=3:B='3'
			elif len(A._charts)==4:B='4'
			elif len(A._charts)>=6:B='6'
			else:B=G
		if B=='3':C=A.layout3()
		elif B=='4':C=A.layout4()
		elif B=='6':C=A.layout6()
		elif B=='2x2':C=A.grid(2,2)
		elif B=='3x3':C=A.grid(3,3)
		elif B==G:import math as D;E=len(A._charts);F=D.ceil(D.sqrt(E));H=D.ceil(E/F);C=A.grid(H,F)
		else:raise ValueError(f"Unknown layout type: {B}")
		if A.env==Environment.STREAMLIT:import streamlit as I;I.components.v1.html(C.embed(max_width=True).html,height=A.height);return
		return C