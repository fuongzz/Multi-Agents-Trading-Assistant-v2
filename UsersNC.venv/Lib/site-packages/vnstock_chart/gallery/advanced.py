_M='{value}%'
_L='Returns %'
_K='dashed'
_J='axis'
_I='none'
_H=True
_G='20px'
_F=None
_E='dark'
_D='border_color'
_C='center'
_B='middle'
_A=False
import pandas as pd
import numpy as np
from typing import Optional,Dict,List,Any
from pyecharts import options as opts
from pyecharts.charts import Line,Bar,Scatter,HeatMap
from pyecharts.globals import ThemeType
from vnstock_chart.core.base import ChartBase
class RoundTripAnalysis(ChartBase):
	def _build(C,*,trades_data:pd.DataFrame,**O):
		I='cumulative_pnl';E='trade_number';B='pnl';A=trades_data.sort_values('exit_date').copy();A[I]=A[B].cumsum();A[E]=range(1,len(A)+1);D=Scatter(init_opts=opts.InitOpts(theme=ThemeType.DARK if C.theme==_E else ThemeType.LIGHT,width=f"{C.width}px",height=f"{C.height}px",bg_color=C.bg_color)).add_xaxis(A[E].tolist());G=[];J=[]
		for(P,F)in A.iterrows():G.append([F[E],F[B]]);J.append(C.bull_color if F[B]>0 else C.bear_color)
		D.add_yaxis(series_name='Trade P&L',y_axis=G,symbol_size=8,label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=_F));H=Line().add_xaxis(A[E].tolist());H.add_yaxis(series_name='Cumulative P&L',y_axis=A[I].tolist(),is_smooth=_H,symbol=_I,linestyle_opts=opts.LineStyleOpts(color=C.text_color,width=3),label_opts=opts.LabelOpts(is_show=_A));D=D.overlap(H);K=(A[B]>0).mean()*100;L=A[A[B]>0][B].mean();M=A[A[B]<0][B].mean();N=abs(A[A[B]>0][B].sum())/abs(A[A[B]<0][B].sum())if len(A[A[B]<0])>0 else np.inf;D.set_global_opts(title_opts=opts.TitleOpts(title='Round-Trip Analysis',subtitle=f"Win Rate: {K:.1f}% | Avg Win: {L:.2f} | Avg Loss: {M:.2f} | PF: {N:.2f}",pos_top=_G,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=18,color=C.text_color)),tooltip_opts=opts.TooltipOpts(formatter='Trade {b}: {c}'),xaxis_opts=opts.AxisOpts(name='Trade Number',name_location=_B,name_gap=30),yaxis_opts=opts.AxisOpts(name='P&L',name_location=_B,name_gap=50));return D
class ExecutionVisualization(ChartBase):
	def _build(A,*,execution_data:pd.DataFrame,**I):E='Slippage (bps)';C='slippage';B=execution_data;F=[A.strftime('%Y-%m-%d %H:%M')for A in B['timestamp']];D=Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis(F).add_yaxis(series_name=E,y_axis=B[C].tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bear_color,border_color=A.border_color if hasattr(A,_D)and A.border_color else _F,border_width=1 if hasattr(A,_D)and A.border_color else 0));G=B[C].mean();H=B[C].max();D.set_global_opts(title_opts=opts.TitleOpts(title='Execution Analysis',subtitle=f"Avg Slippage: {G:.1f}bps | Max: {H:.1f}bps",pos_top=_G,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=18,color=A.text_color)),tooltip_opts=opts.TooltipOpts(formatter='{b}<br/>Slippage: {c}bps'),yaxis_opts=opts.AxisOpts(name=E,name_location=_B,name_gap=50));return D
class CapacityAnalysis(ChartBase):
	def _build(A,*,capacity_data:pd.DataFrame,**H):
		G='Sharpe Ratio';F='sharpe';E='aum_size';B=capacity_data;C=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis(B[E].tolist()).add_yaxis(series_name='Annual Returns',y_axis=B['returns'].tolist(),is_smooth=_H,symbol=_I,linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=3),label_opts=opts.LabelOpts(is_show=_A))
		if F in B.columns:D=Line().add_xaxis(B[E].tolist());D.add_yaxis(series_name=G,y_axis=B[F].tolist(),is_smooth=_H,symbol=_I,linestyle_opts=opts.LineStyleOpts(color=A.text_color,width=2,type_=_K),label_opts=opts.LabelOpts(is_show=_A),yaxis_index=1);C=C.overlap(D);C.set_global_opts(yaxis_opts=[opts.AxisOpts(name=_L,position='left',axislabel_opts=opts.LabelOpts(formatter=_M)),opts.AxisOpts(name=G,position='right',axislabel_opts=opts.LabelOpts(formatter='{value:.2f}'))])
		C.set_global_opts(title_opts=opts.TitleOpts(title='Strategy Capacity Analysis',subtitle='Performance vs AUM Size',pos_top=_G,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=18,color=A.text_color)),tooltip_opts=opts.TooltipOpts(trigger=_J),xaxis_opts=opts.AxisOpts(name='AUM Size ($M)',name_location=_B,name_gap=30));return C
class MonteCarloResults(ChartBase):
	def _build(A,*,simulation_results:np.ndarray,percentiles:List[float]=[5,25,50,75,95],**M):
		I='p50';H='#FFA500';C=simulation_results;J=range(C.shape[1]);E={}
		for F in percentiles:E[f"p{F}"]=np.percentile(C,F,axis=0)
		D=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis(list(J));G=[A.bear_color,H,A.text_color,H,A.bull_color]
		for(K,(B,L))in enumerate(E.items()):D.add_yaxis(series_name=f"{B.upper()} Percentile",y_axis=L.tolist(),is_smooth=_H,symbol=_I,linestyle_opts=opts.LineStyleOpts(color=G[K%len(G)],width=2 if B in[I]else 1,type_='solid'if B==I else _K),label_opts=opts.LabelOpts(is_show=_A),areastyle_opts=opts.AreaStyleOpts(opacity=.1)if B in['p25','p75']else _F)
		D.set_global_opts(title_opts=opts.TitleOpts(title='Monte Carlo Simulation Results',subtitle=f"{C.shape[0]} simulations",pos_top=_G,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=18,color=A.text_color)),tooltip_opts=opts.TooltipOpts(trigger=_J),xaxis_opts=opts.AxisOpts(name='Time Steps',name_location=_B,name_gap=30),yaxis_opts=opts.AxisOpts(name='Portfolio Value',name_location=_B,name_gap=50),legend_opts=opts.LegendOpts(pos_top='60px',pos_left=_C));return D
class WalkForwardResults(ChartBase):
	def _build(A,*,walkforward_data:pd.DataFrame,**H):E='out_sample';D='in_sample';B=walkforward_data;F=B['period'].tolist();C=Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_E else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis(F).add_yaxis(series_name='In-Sample Returns',y_axis=B[D].tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color,opacity=.7,border_color=A.border_color if hasattr(A,_D)and A.border_color else _F,border_width=1 if hasattr(A,_D)and A.border_color else 0)).add_yaxis(series_name='Out-of-Sample Returns',y_axis=B[E].tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bear_color,opacity=.7,border_color=A.border_color if hasattr(A,_D)and A.border_color else _F,border_width=1 if hasattr(A,_D)and A.border_color else 0));G=B[D].corr(B[E]);C.set_global_opts(title_opts=opts.TitleOpts(title='Walk-Forward Analysis',subtitle=f"IS vs OOS Correlation: {G:.3f}",pos_top=_G,pos_left=_C,title_textstyle_opts=opts.TextStyleOpts(font_size=18,color=A.text_color)),tooltip_opts=opts.TooltipOpts(trigger=_J,formatter='{b}<br/>{a}: {c}%'),xaxis_opts=opts.AxisOpts(name='Period',name_location=_B,name_gap=30),yaxis_opts=opts.AxisOpts(name=_L,name_location=_B,name_gap=50,axislabel_opts=opts.LabelOpts(formatter=_M)),legend_opts=opts.LegendOpts(pos_top='60px',pos_left=_C));return C