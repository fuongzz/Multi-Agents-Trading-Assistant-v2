from pyecharts.charts import Bar
from pyecharts import options as opts
from vnstock_chart.core.base import ChartBase
from vnstock_chart.utils import validate_xy
class BarChart(ChartBase):
	def _build(A,*,x=None,y=None,name:str=None,show_title:bool=True,show_legend:bool=False)->Bar:
		validate_xy(x,y);C={'bg_color':A.bg_color}
		if A.width:C['width']=f"{A.width}px"
		if A.height:C['height']=f"{A.height}px"
		B=Bar(init_opts=opts.InitOpts(**C));B.add_xaxis(x);B.add_yaxis(series_name=name or'Value',y_axis=y,label_opts=opts.LabelOpts(is_show=False),itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color));B.set_global_opts(title_opts=opts.TitleOpts(title=A.title,pos_top='20px',pos_left='center',title_textstyle_opts=opts.TextStyleOpts(font_size=22,color=A.text_color),is_show=show_title),legend_opts=opts.LegendOpts(is_show=show_legend));return B