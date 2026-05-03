import sys
import os
from enum import Enum,auto
class Environment(Enum):COLAB=auto();NOTEBOOK=auto();STREAMLIT=auto();TERMINAL=auto();VSCODE_NOTEBOOK=auto()
class EnvDetector:
	@staticmethod
	def detect()->Environment:
		if'google.colab'in sys.modules:return Environment.COLAB
		try:
			A=__import__('IPython').get_ipython()
			if A:
				if A.__class__.__name__=='ZMQInteractiveShell':return Environment.NOTEBOOK
				if'ipykernel'in A.__module__ or'jupyter'in A.__module__:return Environment.VSCODE_NOTEBOOK
		except Exception:pass
		if os.getenv('STREAMLIT_SERVER_RUNS')=='true':return Environment.STREAMLIT
		return Environment.TERMINAL