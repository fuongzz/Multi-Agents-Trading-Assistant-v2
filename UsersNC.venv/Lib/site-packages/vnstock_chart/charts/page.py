_A='simple'
from pyecharts.charts import Page
from pyecharts import options as opts
from vnstock_chart.core.base import ChartBase
class PageChart(ChartBase):
	def _build(D,*,charts:list,layout:str=_A):
		B=Page.SimplePageLayout if layout==_A else Page.DraggablePageLayout;A=Page(layout=B)
		for C in charts:A.add(C._chart)
		return A