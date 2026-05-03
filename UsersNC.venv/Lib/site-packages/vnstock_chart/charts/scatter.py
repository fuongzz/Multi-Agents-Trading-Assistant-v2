from pyecharts.charts import Scatter
from pyecharts import options as opts
from vnstock_chart.core.base import ChartBase
from vnstock_chart.utils import validate_xy
class ScatterChart(ChartBase):
	def _build(A,*,x,y,name:str=None,symbol_size:int=10):validate_xy(x,y);C={'bg_color':A.bg_color,'width':f"{A.width}px",'height':f"{A.height}px"};B=Scatter(init_opts=opts.InitOpts(**C));B.add_xaxis(x);B.add_yaxis(series_name=name or A.title,y_axis=y,symbol_size=symbol_size,itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color));return B