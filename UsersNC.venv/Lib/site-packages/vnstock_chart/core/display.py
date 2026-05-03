import os
import uuid
import time
import shutil
import tempfile
import atexit
from IPython.display import HTML,display
from vnstock_chart.core.env import EnvDetector,Environment
class DisplayManager:
	temp_dirs=[]
	@classmethod
	def cleanup_temp_dirs(A):
		for B in A.temp_dirs:
			try:
				if os.path.exists(B):shutil.rmtree(B,ignore_errors=True)
			except Exception:pass
		A.temp_dirs=[]
	def __init__(A):A.env=EnvDetector.detect();A.temp_dir=tempfile.mkdtemp(prefix='vnchart_');DisplayManager.temp_dirs.append(A.temp_dir)
	def show(A,chart,height:int=400):
		B=chart
		if A.env==Environment.COLAB:import panel as C;C.extension('echarts');return C.pane.ECharts(B)
		if A.env==Environment.NOTEBOOK or A.env==Environment.VSCODE_NOTEBOOK:
			E=f"{int(time.time())}_{uuid.uuid4().hex[:8]}";D=os.path.join(A.temp_dir,f"chart_{E}.html");B.render(D)
			with open(D,'r',encoding='utf-8')as F:G=F.read()
			display(HTML(G));return
		if A.env==Environment.STREAMLIT:import streamlit as H;I=B.get_embed_html();H.components.v1.html(I,height=height);return
		return B.render()
atexit.register(DisplayManager.cleanup_temp_dirs)