_E='file://'
_D='center'
_C=False
_B=True
_A=None
from abc import ABC,abstractmethod
from typing import Any,Dict,Optional
from pyecharts import options as opts
from vnstock_chart.themes import PALETTES,DEFAULT_COLOR_CATEGORY,DEFAULT_FONT_FAMILY,DEFAULT_FONT_SIZE,DEFAULT_WIDTH,DEFAULT_HEIGHT,DEFAULT_GRID_WIDTH,DEFAULT_GRID_OPACITY,CHART_SIZE_PRESETS,DEFAULT_SIZE_PRESET
from vnstock_chart.core.env import EnvDetector
from vnstock_chart.core.display import DisplayManager
from vnstock_chart.core.exporter import Exporter
class ChartBase(ABC):
	def __init__(A,*,theme:str=_A,color_category:str=_A,title:str='',width:int=_A,height:int=_A,size_preset:str=_A,watermark:bool=_B,**H):
		G='light';E=height;D=width;C=color_category;A.env=EnvDetector.detect();A.theme=(theme or'dark').lower()
		if C is _A:
			if A.theme==G:A.category=G
			else:A.category=DEFAULT_COLOR_CATEGORY
		else:A.category=C.lower()
		A.title=title
		if D is not _A or E is not _A:A.width=D or DEFAULT_WIDTH;A.height=E or DEFAULT_HEIGHT
		else:I=size_preset or DEFAULT_SIZE_PRESET;F=CHART_SIZE_PRESETS.get(I,CHART_SIZE_PRESETS[DEFAULT_SIZE_PRESET]);A.width=F['width'];A.height=F['height']
		A.watermark_on=watermark;B=PALETTES.get(A.category,PALETTES[DEFAULT_COLOR_CATEGORY]);A.bull_color=B['bull'];A.bear_color=B['bear'];A.bg_color=B['bg'];A.text_color=B['text'];A.border_color=B.get('border',_A);A.indicator_color=B.get('indicator_color',A.bull_color);A.border_width=.5 if'paper'in A.category else 1;A._chart=A._build(**H)
		try:A._chart.set_global_opts(**A._common_opts())
		except AttributeError:pass
		if A.watermark_on:
			try:A._chart.set_global_opts(graphic_opts=A._watermark())
			except AttributeError:pass
		A._disp=DisplayManager();A._export=Exporter(A._chart)
	def _update_chart(A,new_chart):A._chart=new_chart;A._export=Exporter(A._chart)
	@abstractmethod
	def _build(self,**A)->Any:raise NotImplementedError('Subclasses must implement _build() to return a pyecharts Chart')
	def _common_opts(A)->Dict[str,Any]:return{'title_opts':opts.TitleOpts(title=A.title,pos_top=20,pos_left=_D,title_textstyle_opts=opts.TextStyleOpts(font_family=DEFAULT_FONT_FAMILY,font_size=int(DEFAULT_FONT_SIZE*1.5),color=A.text_color)),'tooltip_opts':opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),'toolbox_opts':opts.ToolboxOpts(is_show=_B,orient='vertical',pos_left='right',feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(),restore=opts.ToolBoxFeatureRestoreOpts(),data_view=opts.ToolBoxFeatureDataViewOpts(),data_zoom=opts.ToolBoxFeatureDataZoomOpts(),magic_type=opts.ToolBoxFeatureMagicTypeOpts(type_=['line','bar','stack','tiled']))),'datazoom_opts':[opts.DataZoomOpts(is_show=_B,type_='inside',xaxis_index=[0,1],range_start=0,range_end=100),opts.DataZoomOpts(is_show=_B,type_='slider',xaxis_index=[0,1],range_start=0,range_end=100)],'legend_opts':opts.LegendOpts(is_show=_C),'xaxis_opts':opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=_B,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY))),'yaxis_opts':opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=_B,linestyle_opts=opts.LineStyleOpts(width=DEFAULT_GRID_WIDTH,opacity=DEFAULT_GRID_OPACITY)))}
	def _watermark(A)->list:return[opts.GraphicImage(graphic_item=opts.GraphicItem(id_='logo',left=_D,top='middle',z=-10,bounding='raw',origin=[.5,.5]),graphic_imagestyle_opts=opts.GraphicImageStyleOpts(image='https://vnstocks.com/img/vnstock_logo_trans_rec_hoz_bw.png',width=180,height=67,opacity=.15))]
	def render(A,auto_open:Optional[bool]=_A):
		E='terminal';B=auto_open;F=A._disp.show(A._chart,height=A.height)
		if B is _A:B=A.env==E
		if B and A.env==E:import tempfile as G;import webbrowser as H;import os;C=G.NamedTemporaryFile(mode='w',suffix='.html',delete=_C);D=C.name;C.close();A._export.to_html(D);H.open(_E+os.path.abspath(D))
		return F
	def to_html(A,path:str,auto_open:bool=_C):
		A._export.to_html(path)
		if auto_open:import webbrowser as B;import os;B.open(_E+os.path.abspath(path))
	def embed(A)->str:return A._export.embed()