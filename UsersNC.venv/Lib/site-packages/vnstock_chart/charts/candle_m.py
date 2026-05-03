_N='file://'
_M='center'
_L='inside'
_K='overlay'
_J='none'
_I='value'
_H='close'
_G='8%'
_F='7%'
_E='dark'
_D='color'
_C=True
_B=False
_A=None
import pandas as pd
import numpy as np
from typing import Any,Dict,List,Optional
from pyecharts import options as opts
from pyecharts.charts import Kline,Line,Bar,Grid
from pyecharts.globals import ThemeType
from vnstock_chart.core.base import ChartBase
from vnstock_chart.themes.palettes import GRADIENT_EMERALD
from vnstock_chart.themes.styling import DEFAULT_GRID_WIDTH,DEFAULT_GRID_OPACITY
class CandleM(ChartBase):
	def __init__(A,**B):
		N='light';M='height';L='width';O=B.pop('theme',_E);D=B.pop('color_category',_A);P=B.pop('title','');E=B.pop(L,_A);F=B.pop(M,_A);Q=B.pop('size_preset',_A);R=B.pop('watermark',_C);from vnstock_chart.core.env import EnvDetector as S;from vnstock_chart.core.display import DisplayManager as T;from vnstock_chart.core.exporter import Exporter;from vnstock_chart.themes import PALETTES as G,DEFAULT_COLOR_CATEGORY as H,DEFAULT_WIDTH as U,DEFAULT_HEIGHT as V,CHART_SIZE_PRESETS as I,DEFAULT_SIZE_PRESET as J;A.env=S.detect();A.theme=(O or _E).lower()
		if D is _A:
			if A.theme==N:A.category=N
			else:A.category=H
		else:A.category=D.lower()
		A.title=P
		if E is not _A or F is not _A:A.width=E or U;A.height=F or V
		else:W=Q or J;K=I.get(W,I[J]);A.width=K[L];A.height=K[M]
		A.watermark_on=R;C=G.get(A.category,G[H]);A.bull_color=C['bull'];A.bear_color=C['bear'];A.bg_color=C['bg'];A.text_color=C['text'];A.border_color=C.get('border',_A);A.indicator_color=C.get('indicator_color',A.bull_color);A._chart=_A;A._disp=T();A._export=_A
	def _build(A,**B):return Grid()
	def build(A,*,df:pd.DataFrame,signals:Optional[pd.DataFrame]=_A,trades:Optional[pd.DataFrame]=_A,indicators:Optional[List[Dict[str,Any]]]=_A,mode:str=_K):
		E=mode;C=trades;B=df;J=A._detect_time_format(pd.Series(B.index));D=B.index.strftime(J).tolist();H=B[['open',_H,'low','high']].values.tolist();K=B['volume'].tolist();A._dates=D;A._df=B;A._signals=signals;A._trades=C;A._indicators=indicators or[];F=[]
		if C is not _A and not C.empty:F=A._generate_trade_markers(C,B)
		if E=='minimal':G=A._build_price_chart(D,H,F)
		else:G=A._build_candlestick_chart(D,H,F,E)
		I=A._build_volume_chart(D,K,B)
		if E=='subplot'and A._indicators:A._chart=A._layout_subplot(G,I,A._indicators)
		else:A._chart=A._layout_basic(G,I)
		from vnstock_chart.core.exporter import Exporter as L;A._export=L(A._chart);return A
	def _generate_trade_markers(B,trades:pd.DataFrame,price_df:pd.DataFrame)->List[opts.MarkPointItem]:
		O='arrow';N='Long';I='exit_date';C=price_df;D=[];E=B._detect_time_format(pd.Series(C.index))
		for(T,A)in trades.iterrows():
			F=pd.to_datetime(A['entry_date'])
			if F in C.index:
				P=A.get('entry_price',C.loc[F,_H]);J=A.get('type',N)
				if J==N:K=O;L=0;M=B.bull_color
				else:K=O;L=180;M=B.bear_color
				E=B._detect_time_format(C.index);D.append(opts.MarkPointItem(name=f"{J} Entry",coord=[F.strftime(E),P],symbol=K,symbol_size=20,symbol_rotate=L,itemstyle_opts=opts.ItemStyleOpts(color=M)))
			if I in A and pd.notna(A[I]):
				G=pd.to_datetime(A[I])
				if G in C.index:Q=A.get('exit_price',C.loc[G,_H]);H=A.get('pnl',0);R=B.bull_color if H>0 else B.bear_color;S='+'if H>0 else'';D.append(opts.MarkPointItem(name=f"Exit ({S}{H:.2f}%)",coord=[G.strftime(E),Q],symbol='circle',symbol_size=15,itemstyle_opts=opts.ItemStyleOpts(color=R)))
		return D
	def _build_candlestick_chart(A,dates:List[str],ohlc:List[List[float]],markers:List[opts.MarkPointItem],mode:str)->Kline:
		F='0px';D=dates;E=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT;B=Kline(init_opts=opts.InitOpts(theme=E,width=f"{A.width}px",height=f"{int(A.height*.65)}px",bg_color=A.bg_color)).add_xaxis(D).add_yaxis(series_name='Price',y_axis=ohlc,itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color,color0=A.bear_color,border_color=A.border_color if A.border_color else A.bull_color,border_color0=A.border_color if A.border_color else A.bear_color),markpoint_opts=opts.MarkPointOpts(data=markers,label_opts=opts.LabelOpts(position=_L,color=A.text_color,font_size=10)));B.set_global_opts(yaxis_opts=opts.AxisOpts(is_scale=_C,splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)),position='right'),xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=_B),splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY))),visualmap_opts=opts.VisualMapOpts(is_show=_B,dimension=2,series_index=5,is_piecewise=_C,pieces=[{_I:1,_D:A.bull_color},{_I:-1,_D:A.bear_color}]))
		if mode==_K and A._indicators:
			for C in A._indicators:G=Line(init_opts=opts.InitOpts(theme=E,width=F,height=F,bg_color='transparent')).add_xaxis(D).add_yaxis(series_name=C['name'],y_axis=C['data'],is_smooth=_C,is_connect_nones=_B,label_opts=opts.LabelOpts(is_show=_B),symbol=_J,linestyle_opts=opts.LineStyleOpts(color=C.get(_D,A.indicator_color),width=2));B=B.overlap(G)
		A._apply_common(B);B.set_global_opts(title_opts=opts.TitleOpts(title=A.title,pos_top='20px',pos_left=_M,title_textstyle_opts=opts.TextStyleOpts(font_size=22,color=A.text_color)),legend_opts=opts.LegendOpts(is_show=_B));return B
	def _build_price_chart(A,dates:List[str],ohlc:List[List[float]],markers:List[opts.MarkPointItem])->Line:C='offset';D=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT;E=[A[1]for A in ohlc];B=Line(init_opts=opts.InitOpts(theme=D,width=f"{A.width}px",height=f"{int(A.height*.65)}px",bg_color=A.bg_color)).add_xaxis(dates).add_yaxis(series_name='Price',y_axis=E,is_smooth=_C,symbol=_J,label_opts=opts.LabelOpts(is_show=_B),areastyle_opts=opts.AreaStyleOpts(opacity=.2,color={'type':'linear','x':0,'y':0,'x2':0,'y2':1,'colorStops':[{C:0,_D:GRADIENT_EMERALD[0]},{C:1,_D:GRADIENT_EMERALD[1]}]}),linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=2),markpoint_opts=opts.MarkPointOpts(data=markers,label_opts=opts.LabelOpts(position=_L,color=A.text_color,font_size=10)));A._apply_common(B);B.set_global_opts(title_opts=opts.TitleOpts(title=A.title,pos_top='20px',pos_left=_M,title_textstyle_opts=opts.TextStyleOpts(font_size=22,color=A.text_color)),legend_opts=opts.LegendOpts(is_show=_B));return B
	def _build_volume_chart(A,dates:List[str],volume:List[float],df:pd.DataFrame)->Bar:
		I='Volume';H='#FFFFFF';G='vol_color';from vnstock_chart.charts.bar import BarChart as J;B=df.copy();B[G]=np.where(B[_H]>=B['open'],A.bull_color,A.bear_color);F=[]
		for(K,C)in zip(volume,B[G]):
			D={_D:C}
			if A.border_color and(C==H or C.upper()==H):D['borderColor']=A.border_color;D['borderWidth']=1
			F.append({_I:K,'itemStyle':D})
		L=J(theme=A.theme,color_category=A.category,title=I,width=A.width,height=int(A.height*.15));E=L._build(x=dates,y=F,name=I,show_title=_B,show_legend=_B);E.set_series_opts(bar_gap='0%',bar_category_gap='5%');E.set_global_opts(xaxis_opts=opts.AxisOpts(type_='category',boundary_gap=_B,axisline_opts=opts.AxisLineOpts(is_on_zero=_B),axistick_opts=opts.AxisTickOpts(is_show=_B),splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)),axislabel_opts=opts.LabelOpts(is_show=_B)),yaxis_opts=opts.AxisOpts(is_scale=_C,splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)),axisline_opts=opts.AxisLineOpts(is_show=_B),axistick_opts=opts.AxisTickOpts(is_show=_B)));return E
	def _layout_basic(B,main:Any,vol:Any)->Grid:A=B._new_grid();A.add(main,grid_opts=opts.GridOpts(pos_left=_F,pos_right=_G,height='60%'));A.add(vol,grid_opts=opts.GridOpts(pos_left=_F,pos_right=_G,pos_top='73%',height='15%'));return A
	def _layout_subplot(A,main:Any,vol:Any,indicators:List[Dict[str,Any]])->Grid:
		B=A._new_grid();B.add(main,grid_opts=opts.GridOpts(pos_left=_F,pos_right=_G,height='50%'));B.add(vol,grid_opts=opts.GridOpts(pos_left=_F,pos_right=_G,pos_top='62%',height='15%'));D=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT
		for(E,C)in enumerate(indicators):F=Line(init_opts=opts.InitOpts(theme=D,width=f"{A.width}px",height=f"{int(A.height*.1)}px",bg_color=A.bg_color)).add_xaxis(A._dates).add_yaxis(series_name=C['name'],y_axis=C['data'],is_smooth=_C,is_connect_nones=_B,label_opts=opts.LabelOpts(is_show=_B),symbol=_J,linestyle_opts=opts.LineStyleOpts(color=C.get(_D,A.indicator_color),width=2));B.add(F,grid_opts=opts.GridOpts(pos_left=_F,pos_right=_G,pos_top=f"{80+E*10}%",height='10%'))
		return B
	def _new_grid(A)->Grid:B=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT;return Grid(init_opts=opts.InitOpts(width=f"{A.width}px",height=f"{A.height}px",theme=B,bg_color=A.bg_color))
	def _detect_time_format(F,time_series)->str:
		C='%Y-%m-%d';B=time_series
		if len(B)<2:return C
		try:
			A=_A
			for D in range(1,min(10,len(B))):
				E=(B.iloc[D]-B.iloc[D-1]).total_seconds()
				if A is _A or E<A:A=E
			if A is _A or A>=86400:return C
			elif A>=3600:return'%Y-%m-%d %H:%M'
			else:return'%Y-%m-%d %H:%M:%S'
		except Exception:return C
	def _apply_common(B,chart:Any):A=B._common_opts().copy();A.pop('title_opts',_A);A.pop('legend_opts',_A);chart.set_global_opts(**A)
	def render(A,auto_open:Optional[bool]=_A):
		E='terminal';B=auto_open
		if A._chart is _A:raise RuntimeError('Chart not built. Call build() before render().')
		F=A._disp.show(A._chart,height=A.height)
		if B is _A:B=A.env==E
		if B and A.env==E:import tempfile as G;import webbrowser as H;import os;C=G.NamedTemporaryFile(mode='w',suffix='.html',delete=_B);D=C.name;C.close();A.to_html(D,auto_open=_B);H.open(_N+os.path.abspath(D))
		return F
	def to_html(A,path:str,auto_open:bool=_B):
		if A._chart is _A:raise RuntimeError('Chart not built. Call build() before to_html().')
		if A._export is _A:from vnstock_chart.core.exporter import Exporter as B;A._export=B(A._chart)
		A._export.to_html(path)
		if auto_open:import webbrowser as C;import os;C.open(_N+os.path.abspath(path))