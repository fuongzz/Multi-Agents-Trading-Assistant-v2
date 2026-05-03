_J='%Y-%m-%d'
_I='middle'
_H='border_color'
_G='weight'
_F='20px'
_E='none'
_D=True
_C='center'
_B='dark'
_A=False
import pandas as pd
import numpy as np
from typing import Optional,Dict,List
from datetime import datetime,timedelta
from pyecharts import options as opts
from pyecharts.charts import Line,Bar,Pie,Scatter
from pyecharts.globals import ThemeType
from vnstock_chart.core.base import ChartBase
class BenchmarkComparison(ChartBase):
	def _build(A,*,portfolio_returns:pd.Series,benchmark_returns:pd.Series,window:int=252,**R):H=benchmark_returns;G=portfolio_returns;F='portfolio';D='benchmark';C=window;E=(1+G).cumprod();I=(1+H).cumprod();B=pd.concat([G,H],axis=1,join='inner');B.columns=[F,D];M=B[F].rolling(C).cov(B[D]);N=B[D].rolling(C).var();O=M/N;S=(B[F].rolling(C).mean()-O*B[D].rolling(C).mean())*252;P=[A.strftime(_J)for A in E.index];J=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_B else ThemeType.LIGHT,width=f"{A.width}px",height=f"{int(A.height*.4)}px",bg_color=A.bg_color)).add_xaxis(P).add_yaxis(series_name='Portfolio',y_axis=E.tolist(),is_smooth=_D,symbol=_E,linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=3),label_opts=opts.LabelOpts(is_show=_A)).add_yaxis(series_name='Benchmark',y_axis=I.tolist(),is_smooth=_D,symbol=_E,linestyle_opts=opts.LineStyleOpts(color=A.text_color,width=2,type_='dashed'),label_opts=opts.LabelOpts(is_show=_A));K=E.iloc[-1]-1;L=I.iloc[-1]-1;Q=K-L;J.set_global_opts(title_opts=opts.TitleOpts(title='Portfolio vs Benchmark',subtitle=f"Excess Return: {Q:.2%} | Portfolio: {K:.2%} | Benchmark: {L:.2%}",pos_top=_F,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color)),tooltip_opts=opts.TooltipOpts(trigger='axis'),yaxis_opts=opts.AxisOpts(name='Cumulative Return',axislabel_opts=opts.LabelOpts(formatter='{value:.1f}x')));return J
class ExposureChart(ChartBase):
	def _build(A,*,exposure_data:pd.DataFrame,**H):
		F='net';E='short';D='long';B=exposure_data;G=[A.strftime(_J)for A in B.index];C=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_B else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis(G)
		if D in B.columns:C.add_yaxis(series_name='Long Exposure',y_axis=B[D].tolist(),is_smooth=_D,symbol=_E,linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=2),label_opts=opts.LabelOpts(is_show=_A),areastyle_opts=opts.AreaStyleOpts(opacity=.3,color=A.bull_color))
		if E in B.columns:C.add_yaxis(series_name='Short Exposure',y_axis=B[E].tolist(),is_smooth=_D,symbol=_E,linestyle_opts=opts.LineStyleOpts(color=A.bear_color,width=2),label_opts=opts.LabelOpts(is_show=_A),areastyle_opts=opts.AreaStyleOpts(opacity=.3,color=A.bear_color))
		if F in B.columns:C.add_yaxis(series_name='Net Exposure',y_axis=B[F].tolist(),is_smooth=_D,symbol=_E,linestyle_opts=opts.LineStyleOpts(color=A.text_color,width=3,type_='solid'),label_opts=opts.LabelOpts(is_show=_A))
		C.set_global_opts(title_opts=opts.TitleOpts(title='Portfolio Exposure',pos_top=_F,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color)),tooltip_opts=opts.TooltipOpts(trigger='axis'),yaxis_opts=opts.AxisOpts(name='Exposure %',axislabel_opts=opts.LabelOpts(formatter='{value}%')),legend_opts=opts.LegendOpts(pos_top='60px',pos_left=_C));return C
class TopPositions(ChartBase):
	def _build(A,*,positions_data:pd.DataFrame,top_n:int=10,chart_type:str='bar',**G):
		E='symbol';D=top_n;B=positions_data.nlargest(D,_G)
		if chart_type=='pie':F=[[A[E],A[_G]]for(B,A)in B.iterrows()];C=Pie(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_B else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add(series_name='Position Weight',data_pair=F,radius=['30%','75%'],label_opts=opts.LabelOpts(formatter='{b}: {d}%'))
		else:C=Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_B else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis(B[E].tolist()).add_yaxis(series_name='Weight %',y_axis=B[_G].tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color,border_color=A.border_color if hasattr(A,_H)and A.border_color else None,border_width=1 if hasattr(A,_H)and A.border_color else 0))
		C.set_global_opts(title_opts=opts.TitleOpts(title=f"Top {D} Positions",pos_top=_F,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color)),tooltip_opts=opts.TooltipOpts(formatter='{b}: {c}%'));return C
class TradeDuration(ChartBase):
	def _build(A,*,trades_data:pd.DataFrame,bins:int=20,**K):F='Number of Trades';C='duration';B=trades_data;B[C]=(pd.to_datetime(B['exit_date'])-pd.to_datetime(B['entry_date'])).dt.days;G,D=np.histogram(B[C].dropna(),bins=bins);H=(D[:-1]+D[1:])/2;E=Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_B else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis([f"{int(A)}d"for A in H]).add_yaxis(series_name=F,y_axis=G.tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color,opacity=.8,border_color=A.border_color if hasattr(A,_H)and A.border_color else None,border_width=1 if hasattr(A,_H)and A.border_color else 0));I=B[C].mean();J=B[C].median();E.set_global_opts(title_opts=opts.TitleOpts(title='Trade Duration Distribution',subtitle=f"Mean: {I:.1f}d | Median: {J:.1f}d",pos_top=_F,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color)),tooltip_opts=opts.TooltipOpts(formatter='{b}: {c} trades'),xaxis_opts=opts.AxisOpts(name='Duration (days)',name_location=_I,name_gap=30),yaxis_opts=opts.AxisOpts(name=F,name_location=_I,name_gap=50));return E
class AllocationChart(ChartBase):
	def _build(A,*,allocation_data:pd.DataFrame,allocation_type:str='sector',**E):B=allocation_type;D=[[A[B],A[_G]]for(C,A)in allocation_data.iterrows()];C=Pie(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_B else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add(series_name=f"{B.title()} Allocation",data_pair=D,radius=['40%','75%'],label_opts=opts.LabelOpts(formatter='{b}\n{d}%',font_size=12));C.set_global_opts(title_opts=opts.TitleOpts(title=f"{B.title()} Allocation",pos_top=_F,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color)),tooltip_opts=opts.TooltipOpts(formatter='{b}: {c}% ({d}%)'),legend_opts=opts.LegendOpts(pos_top=_I,pos_left='80%',orient='vertical'));return C