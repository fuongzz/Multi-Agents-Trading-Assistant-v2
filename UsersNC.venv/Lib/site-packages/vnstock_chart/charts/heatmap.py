from pyecharts.charts import HeatMap
from pyecharts import options as opts
from vnstock_chart.core.base import ChartBase
class HeatmapChart(ChartBase):
	def _build(A,*,x,y,value,min_value:float=None,max_value:float=None,name:str=None):C={'bg_color':A.bg_color,'width':f"{A.width}px",'height':f"{A.height}px"};B=HeatMap(init_opts=opts.InitOpts(**C));B.add_xaxis(x);B.add_yaxis(series_name=name or A.title,yaxis_data=y,value=value,label_opts=opts.LabelOpts(is_show=True,color=A.text_color));B.set_global_opts(visualmap_opts=opts.VisualMapOpts(min_=min_value,max_=max_value,is_calculable=True));return B