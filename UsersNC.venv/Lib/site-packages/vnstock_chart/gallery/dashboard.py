_K='Trading Analytics'
_J='neutral'
_I='Risk Analytics'
_H='paper_bear'
_G='paper_bull'
_F='light'
_E='400px'
_D='1400px'
_C='50%'
_B='0%'
_A=None
import pandas as pd
import numpy as np
from typing import Optional,Dict,List,Any
from pyecharts.charts import Page,Tab,Grid
from pyecharts import options as opts
from.backtest import EquityCurve,ReturnsHistogram,MonthlyReturnsHeatmap,RollingMetrics
from.performance import BenchmarkComparison,ExposureChart,TopPositions,TradeDuration,AllocationChart
from.advanced import RoundTripAnalysis,ExecutionVisualization,CapacityAnalysis,MonteCarloResults,WalkForwardResults
class BacktestDashboard:
	def __init__(A,theme:str=_F,color_category:str=_G,title:str='Backtest Analysis Dashboard'):A.theme=theme;A.color_category=color_category;A.title=title
	def build(A,*,equity_data:pd.Series,returns:pd.Series,trades:Optional[pd.DataFrame]=_A,benchmark:Optional[pd.Series]=_A,positions:Optional[pd.DataFrame]=_A)->Page:
		E=positions;D=returns;C=trades;B=Page(page_title=A.title);H=EquityCurve(theme=A.theme,color_category=A.color_category,width=1400,height=600);B.add(H._build(equity_data=equity_data,benchmark=benchmark));F=Grid(init_opts=opts.InitOpts(width=_D,height=_E));I=ReturnsHistogram(theme=A.theme,color_category=A.color_category,width=700,height=400);J=MonthlyReturnsHeatmap(theme=A.theme,color_category=A.color_category,width=700,height=400);F.add(I._build(returns=D),grid_opts=opts.GridOpts(pos_left=_B,pos_right=_C));F.add(J._build(returns=D),grid_opts=opts.GridOpts(pos_left=_C,pos_right=_B));B.add(F);K=RollingMetrics(theme=A.theme,color_category=A.color_category,width=1400,height=400);B.add(K._build(returns=D))
		if C is not _A and not C.empty:G=Grid(init_opts=opts.InitOpts(width=_D,height=_E));L=RoundTripAnalysis(theme=A.theme,color_category=A.color_category,width=700,height=400);M=TradeDuration(theme=A.theme,color_category=A.color_category,width=700,height=400);G.add(L._build(trades_data=C),grid_opts=opts.GridOpts(pos_left=_B,pos_right=_C));G.add(M._build(trades_data=C),grid_opts=opts.GridOpts(pos_left=_C,pos_right=_B));B.add(G)
		if E is not _A and not E.empty:N=TopPositions(theme=A.theme,color_category=A.color_category,width=1400,height=400);B.add(N._build(positions_data=E,chart_type='bar'))
		return B
	def render(A,path:str='backtest_dashboard.html'):
		if hasattr(A,'_page'):A._page.render(path)
class PerformancePage:
	def __init__(A,theme:str=_F,color_category:str=_G,title:str='Performance Analytics'):A.theme=theme;A.color_category=color_category;A.title=title
	def build(A,*,portfolio_returns:pd.Series,benchmark_returns:pd.Series,exposure_data:Optional[pd.DataFrame]=_A,positions:Optional[pd.DataFrame]=_A,allocation:Optional[pd.DataFrame]=_A)->Page:
		F=positions;D=allocation;C=exposure_data;B=Page(page_title=A.title);G=BenchmarkComparison(theme=A.theme,color_category=A.color_category,width=1400,height=500);B.add(G._build(portfolio_returns=portfolio_returns,benchmark_returns=benchmark_returns))
		if C is not _A or D is not _A:
			E=Grid(init_opts=opts.InitOpts(width=_D,height=_E))
			if C is not _A:H=ExposureChart(theme=A.theme,color_category=A.color_category,width=700,height=400);E.add(H._build(exposure_data=C),grid_opts=opts.GridOpts(pos_left=_B,pos_right=_C))
			if D is not _A:I=AllocationChart(theme=A.theme,color_category=A.color_category,width=700,height=400);E.add(I._build(allocation_data=D),grid_opts=opts.GridOpts(pos_left=_C,pos_right=_B))
			B.add(E)
		if F is not _A:J=TopPositions(theme=A.theme,color_category=A.color_category,width=1400,height=400);B.add(J._build(positions_data=F,chart_type='pie'))
		return B
class RiskPage:
	def __init__(A,theme:str=_F,color_category:str=_H,title:str=_I):A.theme=theme;A.color_category=color_category;A.title=title
	def build(A,*,equity_data:pd.Series,returns:pd.Series,monte_carlo:Optional[np.ndarray]=_A,capacity_data:Optional[pd.DataFrame]=_A)->Page:
		G=returns;D=capacity_data;C=monte_carlo;B=Page(page_title=A.title);H=EquityCurve(theme=A.theme,color_category=A.color_category,width=1400,height=500);B.add(H._build(equity_data=equity_data));E=Grid(init_opts=opts.InitOpts(width=_D,height=_E));I=ReturnsHistogram(theme=A.theme,color_category=A.color_category,width=700,height=400);J=RollingMetrics(theme=A.theme,color_category=A.color_category,width=700,height=400);E.add(I._build(returns=G),grid_opts=opts.GridOpts(pos_left=_B,pos_right=_C));E.add(J._build(returns=G),grid_opts=opts.GridOpts(pos_left=_C,pos_right=_B));B.add(E)
		if C is not _A or D is not _A:
			F=Grid(init_opts=opts.InitOpts(width=_D,height=_E))
			if C is not _A:K=MonteCarloResults(theme=A.theme,color_category=A.color_category,width=700,height=400);F.add(K._build(simulation_results=C),grid_opts=opts.GridOpts(pos_left=_B,pos_right=_C))
			if D is not _A:L=CapacityAnalysis(theme=A.theme,color_category=A.color_category,width=700,height=400);F.add(L._build(capacity_data=D),grid_opts=opts.GridOpts(pos_left=_C,pos_right=_B))
			B.add(F)
		return B
class TradingPage:
	def __init__(A,theme:str=_F,color_category:str=_J,title:str=_K):A.theme=theme;A.color_category=color_category;A.title=title
	def build(A,*,trades:pd.DataFrame,execution_data:Optional[pd.DataFrame]=_A,walkforward_data:Optional[pd.DataFrame]=_A)->Page:
		F=walkforward_data;E=execution_data;D=trades;B=Page(page_title=A.title);G=RoundTripAnalysis(theme=A.theme,color_category=A.color_category,width=1400,height=500);B.add(G._build(trades_data=D));C=Grid(init_opts=opts.InitOpts(width=_D,height=_E));H=TradeDuration(theme=A.theme,color_category=A.color_category,width=700,height=400);C.add(H._build(trades_data=D),grid_opts=opts.GridOpts(pos_left=_B,pos_right=_C))
		if E is not _A:I=ExecutionVisualization(theme=A.theme,color_category=A.color_category,width=700,height=400);C.add(I._build(execution_data=E),grid_opts=opts.GridOpts(pos_left=_C,pos_right=_B))
		B.add(C)
		if F is not _A:J=WalkForwardResults(theme=A.theme,color_category=A.color_category,width=1400,height=400);B.add(J._build(walkforward_data=F))
		return B
class CompleteDashboard:
	def __init__(A,theme:str=_F,color_category:str=_G,title:str='Complete Analytics Dashboard'):A.theme=theme;A.color_category=color_category;A.title=title
	def build(B,*,equity_data:pd.Series,returns:pd.Series,benchmark_returns:pd.Series,trades:pd.DataFrame,**A)->Tab:H='positions';G=trades;F=benchmark_returns;E=equity_data;D=returns;C=Tab();I=BacktestDashboard(theme=B.theme,color_category=B.color_category);J=I.build(equity_data=E,returns=D,trades=G,benchmark=F,positions=A.get(H));C.add(J,'Backtest Analysis');K=PerformancePage(theme=B.theme,color_category=B.color_category);L=K.build(portfolio_returns=D,benchmark_returns=F,exposure_data=A.get('exposure_data'),positions=A.get(H),allocation=A.get('allocation'));C.add(L,'Performance');M=RiskPage(theme=B.theme,color_category=_H);N=M.build(equity_data=E,returns=D,monte_carlo=A.get('monte_carlo'),capacity_data=A.get('capacity_data'));C.add(N,_I);O=TradingPage(theme=B.theme,color_category=_J);P=O.build(trades=G,execution_data=A.get('execution_data'),walkforward_data=A.get('walkforward_data'));C.add(P,_K);return C
	def render(A,path:str='complete_dashboard.html'):
		if hasattr(A,'_tab'):A._tab.render(path)