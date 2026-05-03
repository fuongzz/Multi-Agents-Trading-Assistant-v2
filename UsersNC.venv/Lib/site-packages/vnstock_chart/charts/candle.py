_K='center'
_J='none'
_I='value'
_H='candle'
_G='dark'
_F='8%'
_E='7%'
_D='color'
_C=True
_B=None
_A=False
import pandas as pd
from typing import Any,Dict,List,Optional
from pyecharts import options as opts
from pyecharts.charts import Kline,Line,Bar,Grid
from pyecharts.globals import ThemeType
from vnstock_chart.core.base import ChartBase
from vnstock_chart.themes.palettes import GRADIENT_EMERALD
from vnstock_chart.themes.styling import DEFAULT_GRID_WIDTH,DEFAULT_GRID_OPACITY
class CandleChart(ChartBase):
	def _build(A,*,df:Optional[pd.DataFrame]=_B,dates:Optional[List[str]]=_B,ohlc:Optional[List[List[float]]]=_B,volume:Optional[List[float]]=_B,indicators:Optional[List[Dict[str,Any]]]=_B,mode:str=_H)->Grid:
		J='subplot';I='overlay';H='price';G='time';F=volume;E=indicators;D=ohlc;C=dates;B=df
		if B is not _B:K=A._detect_time_format(B[G]);C=B[G].dt.strftime(K).tolist();D=B[['open','close','low','high']].values.tolist();F=B['volume'].tolist()
		A._last_dates=C or[];E=E or[]
		if C is _B or D is _B or F is _B:return A._new_grid()
		L={H:A._price_area_layer,_H:A._kline_layer,I:A._kline_layer,J:A._kline_layer}[mode](C,D,E);M=A._volume_layer(C,F,D);return{H:A._layout_basic,_H:A._layout_basic,I:A._layout_basic,J:A._layout_subplot}[mode](L,M,E)
	def _apply_common(C,chart:Any,hide_title:bool=_A,hide_legend:bool=_A):B=chart;A=C._common_opts().copy();A.pop('title_opts',_B);A.pop('legend_opts',_B);B.set_global_opts(**A);B.set_global_opts(legend_opts=opts.LegendOpts(is_show=_A))
	def _kline_layer(A,dates:List[str],ohlc:List[List[float]],indicators:List[Dict[str,Any]])->Kline:
		F='0px';D=dates;E=ThemeType.DARK if A.theme==_G else ThemeType.LIGHT;B=Kline(init_opts=opts.InitOpts(theme=E,width=f"{A.width}px",height=f"{int(A.height*.65)}px",bg_color=A.bg_color)).add_xaxis(D).add_yaxis(series_name=A.title,y_axis=ohlc,itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color,color0=A.bear_color,border_color=A.border_color if A.border_color else A.bull_color,border_color0=A.border_color if A.border_color else A.bear_color,border_width=A.border_width));B.set_global_opts(yaxis_opts=opts.AxisOpts(is_scale=_C,splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)),position='right'),xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=_A),splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY))),visualmap_opts=opts.VisualMapOpts(is_show=_A,dimension=2,series_index=5,is_piecewise=_C,pieces=[{_I:1,_D:A.bull_color},{_I:-1,_D:A.bear_color}]))
		for C in indicators:G=Line(init_opts=opts.InitOpts(theme=E,width=F,height=F,bg_color='transparent')).add_xaxis(D).add_yaxis(series_name=C['name'],y_axis=C['data'],is_smooth=_C,is_connect_nones=_A,label_opts=opts.LabelOpts(is_show=_A),symbol=_J,linestyle_opts=opts.LineStyleOpts(color=C.get(_D,A.indicator_color),width=2));B=B.overlap(G)
		A._apply_common(B);B.set_global_opts(title_opts=opts.TitleOpts(title=A.title,pos_top='20px',pos_left=_K,title_textstyle_opts=opts.TextStyleOpts(font_size=22,color=A.text_color)),legend_opts=opts.LegendOpts(is_show=_A));return B
	def _price_area_layer(A,dates:List[str],ohlc:List[List[float]],_:Any)->Line:C='offset';D=ThemeType.DARK if A.theme==_G else ThemeType.LIGHT;E=[A[1]for A in ohlc];B=Line(init_opts=opts.InitOpts(theme=D,width=f"{A.width}px",height=f"{int(A.height*.65)}px",bg_color=A.bg_color)).add_xaxis(dates).add_yaxis(series_name=A.title,y_axis=E,is_smooth=_C,symbol=_J,label_opts=opts.LabelOpts(is_show=_A),areastyle_opts=opts.AreaStyleOpts(opacity=.2,color={'type':'linear','x':0,'y':0,'x2':0,'y2':1,'colorStops':[{C:0,_D:GRADIENT_EMERALD[0]},{C:1,_D:GRADIENT_EMERALD[1]}]}),linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=2));A._apply_common(B,hide_legend=_C);B.set_global_opts(title_opts=opts.TitleOpts(title=A.title,pos_top='20px',pos_left=_K,title_textstyle_opts=opts.TextStyleOpts(font_size=22,color=A.text_color)),legend_opts=opts.LegendOpts(is_show=_A));return B
	def _volume_layer(A,dates:List[str],volume:List[float],ohlc:List[List[float]])->Bar:
		H='Volume';G='#FFFFFF';from vnstock_chart.charts.bar import BarChart as I;E=[]
		for(F,J)in enumerate(volume):
			K=ohlc[F][0];L=ohlc[F][1];B=A.bull_color if L>=K else A.bear_color;C={_D:B}
			if A.border_color and(B==G or B.upper()==G):C['borderColor']=A.border_color;C['borderWidth']=A.border_width
			E.append({_I:J,'itemStyle':C})
		M=I(theme=A.theme,color_category=A.category,title=H,width=A.width,height=int(A.height*.15));D=M._build(x=dates,y=E,name=H,show_title=_A,show_legend=_A);D.set_series_opts(bar_gap='0%',bar_category_gap='5%');D.set_global_opts(xaxis_opts=opts.AxisOpts(type_='category',boundary_gap=_A,axisline_opts=opts.AxisLineOpts(is_on_zero=_A),axistick_opts=opts.AxisTickOpts(is_show=_A),splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)),axislabel_opts=opts.LabelOpts(is_show=_A)),yaxis_opts=opts.AxisOpts(is_scale=_C,splitline_opts=opts.SplitLineOpts(is_show=_C,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)),axisline_opts=opts.AxisLineOpts(is_show=_A),axistick_opts=opts.AxisTickOpts(is_show=_A)));return D
	def _layout_basic(B,main:Any,vol:Any,__:Any)->Grid:A=B._new_grid();A.add(main,grid_opts=opts.GridOpts(pos_left=_E,pos_right=_F,height='55%'));A.add(vol,grid_opts=opts.GridOpts(pos_left=_E,pos_right=_F,pos_top='70%',height='15%'));return A
	def _layout_subplot(A,main:Any,vol:Any,indicators:List[Dict[str,Any]])->Grid:
		B=A._new_grid();B.add(main,grid_opts=opts.GridOpts(pos_left=_E,pos_right=_F,height='45%'));B.add(vol,grid_opts=opts.GridOpts(pos_left=_E,pos_right=_F,pos_top='58%',height='15%'));D=ThemeType.DARK if A.theme==_G else ThemeType.LIGHT
		for(E,C)in enumerate(indicators):F=Line(init_opts=opts.InitOpts(theme=D,width=f"{A.width}px",height=f"{int(A.height*.1)}px",bg_color=A.bg_color)).add_xaxis(A._last_dates).add_yaxis(series_name=C['name'],y_axis=C['data'],is_smooth=_C,is_connect_nones=_A,label_opts=opts.LabelOpts(is_show=_A),symbol=_J,linestyle_opts=opts.LineStyleOpts(color=C.get(_D,A.indicator_color)));B.add(F,grid_opts=opts.GridOpts(pos_left=_E,pos_right=_F,pos_top=f"{80+E*10}%",height='10%'))
		return B
	def _new_grid(A)->Grid:B=ThemeType.DARK if A.theme==_G else ThemeType.LIGHT;C=Grid(init_opts=opts.InitOpts(width=f"{A.width}px",height=f"{A.height}px",theme=B,bg_color=A.bg_color));return C
	def _detect_time_format(F,time_series):
		C='%Y-%m-%d';B=time_series
		if len(B)<2:return C
		try:
			A=_B
			for D in range(1,min(10,len(B))):
				E=(B.iloc[D]-B.iloc[D-1]).total_seconds()
				if A is _B or E<A:A=E
			if A is _B or A>=86400:return C
			elif A>=3600:return'%Y-%m-%d %H:%M'
			else:return'%Y-%m-%d %H:%M:%S'
		except Exception:return C