_K='border_color'
_J='inside'
_I='%Y-%m-%d'
_H='20px'
_G='none'
_F='middle'
_E='center'
_D='dark'
_C=True
_B=None
_A=False
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any,Dict,List,Optional,Union
from pyecharts import options as opts
from pyecharts.charts import Line,Bar,HeatMap,Grid,Page
from pyecharts.globals import ThemeType
from vnstock_chart.core.base import ChartBase
from vnstock_chart.charts.line import LineChart
from vnstock_chart.charts.bar import BarChart
from vnstock_chart.charts.heatmap import HeatmapChart
class EquityCurve(ChartBase):
	def __init__(A,**B):super().__init__(**B);A.watermark_on=_C;A._chart=_B
	def _build(A,**B):from pyecharts.charts import Grid;return Grid()
	def build(A,*,equity_data:pd.Series,drawdown_data:Optional[pd.Series]=_B,benchmark:Optional[pd.Series]=_B,show_drawdown:bool=_C,**B):C=A._build_chart(equity_data=equity_data,drawdown_data=drawdown_data,benchmark=benchmark,show_drawdown=show_drawdown,**B);A._update_chart(C);return A
	def _build_chart(A,*,equity_data:pd.Series,drawdown_data:Optional[pd.Series]=_B,benchmark:Optional[pd.Series]=_B,show_drawdown:bool=_C,**N):
		K='Portfolio Value';G=show_drawdown;F=benchmark;D=drawdown_data;B=equity_data
		if D is _B and G:H=B.expanding().max();D=(B-H)/H*100
		I=[A.strftime(_I)for A in B.index];C=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,width=f"{A.width}px",height=f"{int(A.height*.7)}px",bg_color=A.bg_color)).add_xaxis(I).add_yaxis(series_name=K,y_axis=B.tolist(),is_smooth=_C,symbol=_G,linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=3),label_opts=opts.LabelOpts(is_show=_A),areastyle_opts=opts.AreaStyleOpts(opacity=.1,color=A.bull_color))
		if F is not _B:L=F.reindex(B.index,method='ffill').tolist();C.add_yaxis(series_name='Benchmark',y_axis=L,is_smooth=_C,symbol=_G,linestyle_opts=opts.LineStyleOpts(color=A.text_color,width=2,type_='dashed'),label_opts=opts.LabelOpts(is_show=_A))
		from pyecharts.commons.utils import JsCode as M;C.set_global_opts(title_opts=opts.TitleOpts(title='Equity Curve',subtitle='Portfolio Performance Over Time',pos_top='30px',pos_left=_E,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color),subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12,color=A.text_color)),legend_opts=opts.LegendOpts(is_show=_A),tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),yaxis_opts=opts.AxisOpts(name=K,name_location=_F,name_gap=50,axislabel_opts=opts.LabelOpts(formatter=M("\n                        function(value) {\n                            if (value >= 1e9) {\n                                return (value/1e9).toFixed(1) + 'B';\n                            }\n                            if (value >= 1e6) {\n                                return (value/1e6).toFixed(1) + 'M';\n                            }\n                            if (value >= 1e3) {\n                                return (value/1e3).toFixed(1) + 'K';\n                            }\n                            return value.toFixed(0);\n                        }\n                    ")),splitline_opts=opts.SplitLineOpts(is_show=_A)),xaxis_opts=opts.AxisOpts(type_='category',boundary_gap=_A,splitline_opts=opts.SplitLineOpts(is_show=_A)),datazoom_opts=[opts.DataZoomOpts(type_=_J,range_start=0,range_end=100),opts.DataZoomOpts(type_='slider',range_start=0,range_end=100)])
		if not G or D is _B:return C
		J=Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,width=f"{A.width}px",height=f"{int(A.height*.3)}px",bg_color=A.bg_color)).add_xaxis(I).add_yaxis(series_name='Drawdown (%)',y_axis=D.tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bear_color,border_color=A.border_color if hasattr(A,_K)and A.border_color else A.bear_color,border_width=1));J.set_global_opts(legend_opts=opts.LegendOpts(is_show=_A),yaxis_opts=opts.AxisOpts(name='Drawdown %',name_location=_F,name_gap=40,axislabel_opts=opts.LabelOpts(formatter='{value}%'),splitline_opts=opts.SplitLineOpts(is_show=_A)),xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=_A),splitline_opts=opts.SplitLineOpts(is_show=_A)))
		if getattr(A,'watermark_on',_A):C.set_global_opts(graphic_opts=[opts.GraphicImage(graphic_item=opts.GraphicItem(id_='logo',left=_E,top='45%',z=-10,bounding='raw',origin=[.5,.5]),graphic_imagestyle_opts=opts.GraphicImageStyleOpts(image='https://vnstocks.com/img/vnstock_logo_trans_rec_hoz_bw.png',width=180,height=67,opacity=.15))])
		E=Grid(init_opts=opts.InitOpts(width=f"{A.width}px",height=f"{A.height}px",theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,bg_color=A.bg_color));E.add(C,grid_opts=opts.GridOpts(pos_bottom='35%'));E.add(J,grid_opts=opts.GridOpts(pos_top='70%'));return E
class ReturnsHistogram(ChartBase):
	def _build(A,*,returns:pd.Series,bins:int=50,show_normal:bool=_C,**L):E='Frequency';B=returns;F,C=np.histogram(B.dropna(),bins=bins);G=(C[:-1]+C[1:])/2;D=Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,width=f"{A.width}px",height=f"{A.height}px",bg_color=A.bg_color)).add_xaxis([f"{A:.2%}"for A in G]).add_yaxis(series_name=E,y_axis=F.tolist(),label_opts=opts.LabelOpts(is_show=_A),itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color,opacity=.7,border_color=A.border_color if hasattr(A,_K)and A.border_color else A.bull_color,border_width=1));H=B.mean();I=B.std();J=B.skew();K=B.kurtosis();D.set_global_opts(title_opts=opts.TitleOpts(title='Returns Distribution',subtitle=f"Mean: {H:.2%} | Std: {I:.2%} | Skew: {J:.2f} | Kurt: {K:.2f}",pos_top=_H,pos_left=_E,title_textstyle_opts=opts.TextStyleOpts(font_size=24,color=A.text_color),subtitle_textstyle_opts=opts.TextStyleOpts(font_size=14,color=A.text_color)),tooltip_opts=opts.TooltipOpts(trigger='axis',formatter='{b}: {c} occurrences'),xaxis_opts=opts.AxisOpts(name='Returns',name_location=_F,name_gap=30),yaxis_opts=opts.AxisOpts(name=E,name_location=_F,name_gap=50,axislabel_opts=opts.LabelOpts(formatter="function(value) {\n                        if (value >= 1000000000) return (value/1000000000).toFixed(1) + 'B';\n                        if (value >= 1000000) return (value/1000000).toFixed(1) + 'M';\n                        if (value >= 1000) return (value/1000).toFixed(1) + 'K';\n                        return value.toFixed(0);\n                    }")));return D
class MonthlyReturnsHeatmap(ChartBase):
	def _build(B,*,returns:pd.Series,**J):
		A=returns.resample('M').apply(lambda x:(1+x).prod()-1);A.index=pd.to_datetime(A.index);D=sorted(A.index.year.unique());G=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];C=[]
		for E in D:
			for(F,K)in enumerate(G,1):
				try:
					H=A[(A.index.year==E)&(A.index.month==F)].iloc[0]if len(A[(A.index.year==E)&(A.index.month==F)])>0 else _B
					if H is not _B:C.append([E-min(D),F-1,round(H*100,2)])
				except:pass
		I=HeatMap(init_opts=opts.InitOpts(theme=ThemeType.DARK if B.theme==_D else ThemeType.LIGHT,width=f"{B.width}px",height=f"{B.height}px",bg_color=B.bg_color)).add_xaxis(G).add_yaxis(series_name='Monthly Returns',yaxis_data=list(map(str,D)),value=C,label_opts=opts.LabelOpts(is_show=_C,position=_J,formatter='{c}%'));I.set_global_opts(title_opts=opts.TitleOpts(title='Monthly Returns Heatmap',pos_top=_H,pos_left=_E,title_textstyle_opts=opts.TextStyleOpts(font_size=24,color=B.text_color)),visualmap_opts=opts.VisualMapOpts(min_=min([A[2]for A in C])if C else-10,max_=max([A[2]for A in C])if C else 10,range_color=[B.bear_color,'#FFFFFF',B.bull_color],orient='horizontal',pos_bottom='10%',pos_left=_E),tooltip_opts=opts.TooltipOpts(formatter='{b} {a}: {c}%'));return I
class RollingMetrics(ChartBase):
	def _build(A,*,returns:pd.Series,window:int=252,risk_free_rate:float=.0,**J):C=returns;B=window;H=C.rolling(B).std()*np.sqrt(252)*100;I=(C.rolling(B).mean()*252-risk_free_rate)/C.rolling(B).std()/np.sqrt(252);E=[A.strftime(_I)for A in C.index];F=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,width=f"{A.width}px",height=f"{int(A.height*.5)}px",bg_color=A.bg_color)).add_xaxis(E).add_yaxis(series_name=f"Rolling Volatility ({B}d)",y_axis=H.tolist(),is_smooth=_C,symbol=_G,linestyle_opts=opts.LineStyleOpts(color=A.bear_color,width=2),label_opts=opts.LabelOpts(is_show=_A));F.set_global_opts(title_opts=opts.TitleOpts(title='Rolling Performance Metrics',subtitle='Volatility and Sharpe Ratio Analysis',pos_top=_H,pos_left=_E,title_textstyle_opts=opts.TextStyleOpts(font_size=20,color=A.text_color),subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12,color=A.text_color)),yaxis_opts=opts.AxisOpts(name='Volatility %',name_location=_F,name_gap=50),xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=_A)));G=Line(init_opts=opts.InitOpts(theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,width=f"{A.width}px",height=f"{int(A.height*.5)}px",bg_color=A.bg_color)).add_xaxis(E).add_yaxis(series_name=f"Rolling Sharpe ({B}d)",y_axis=I.tolist(),is_smooth=_C,symbol=_G,linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=2),label_opts=opts.LabelOpts(is_show=_A));G.set_global_opts(title_opts=opts.TitleOpts(is_show=_A),yaxis_opts=opts.AxisOpts(name='Sharpe Ratio',name_location=_F,name_gap=50));D=Grid(init_opts=opts.InitOpts(width=f"{A.width}px",height=f"{A.height}px",theme=ThemeType.DARK if A.theme==_D else ThemeType.LIGHT,bg_color=A.bg_color));D.add(F,grid_opts=opts.GridOpts(pos_bottom='55%'));D.add(G,grid_opts=opts.GridOpts(pos_top='50%'));return D
class BacktestDashboard:
	def __init__(A,theme:str='light',color_category:str='paper_bull',width:int=1400,height:int=1000):A.theme=theme;A.color_category=color_category;A.width=width;A.height=height
	def build(A,*,equity_data:pd.Series,returns:pd.Series,trades:Optional[pd.DataFrame]=_B,benchmark:Optional[pd.Series]=_B)->Page:B=returns;D=EquityCurve(theme=A.theme,color_category=A.color_category,width=A.width,height=int(A.height*.4));E=ReturnsHistogram(theme=A.theme,color_category=A.color_category,width=int(A.width*.5),height=int(A.height*.3),returns=B);F=MonthlyReturnsHeatmap(theme=A.theme,color_category=A.color_category,width=int(A.width*.5),height=int(A.height*.3),returns=B);G=RollingMetrics(theme=A.theme,color_category=A.color_category,width=A.width,height=int(A.height*.3),returns=B);H=D.build(equity_data=equity_data,benchmark=benchmark)._chart;I=E._chart;J=F._chart;K=G._chart;C=Page(page_title='Backtest Analysis Dashboard',layout=Page.DraggablePageLayout);C.add(H,I,J,K);A._page=C;return C
	def render(A,path:str='backtest_dashboard.html'):
		if hasattr(A,'_page'):A._page.render(path)
		else:raise RuntimeError('Build dashboard first using build() method')
	def to_html(A,path:str):A.render(path)