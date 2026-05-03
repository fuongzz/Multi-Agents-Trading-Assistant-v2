from pyecharts.charts import Boxplot
from pyecharts import options as opts
from vnstock_chart.core.base import ChartBase
class BoxplotChart(ChartBase):
	def _build(A,*,x,y,name:str=None):C={'bg_color':A.bg_color,'width':f"{A.width}px",'height':f"{A.height}px"};B=Boxplot(init_opts=opts.InitOpts(**C));B.add_xaxis(x);B.add_yaxis(series_name=name or A.title,y_axis=y);return B