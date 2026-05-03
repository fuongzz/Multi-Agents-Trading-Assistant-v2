from pyecharts.charts import Line
from pyecharts import options as opts
from vnstock_chart.core.base import ChartBase
from vnstock_chart.utils import validate_xy
class LineChart(ChartBase):
	def _build(A,*,x,y,name:str=None)->Line:
		validate_xy(x,y);B={'bg_color':A.bg_color}
		if A.width:B['width']=f"{A.width}px"
		if A.height:B['height']=f"{A.height}px"
		C=Line(init_opts=opts.InitOpts(**B));C.add_xaxis(x);C.add_yaxis(series_name=name or A.title,y_axis=y,linestyle_opts=opts.LineStyleOpts(color=A.bull_color,width=2),itemstyle_opts=opts.ItemStyleOpts(color=A.bull_color),label_opts=opts.LabelOpts(is_show=False),symbol='none',is_smooth=True);return C