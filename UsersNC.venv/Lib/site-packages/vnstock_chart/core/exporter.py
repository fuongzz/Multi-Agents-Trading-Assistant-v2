class Exporter:
	def __init__(A,chart):A.chart=chart
	def to_html(A,path:str):A.chart.render(path)
	def embed(A)->str:return A.chart.render_embed()