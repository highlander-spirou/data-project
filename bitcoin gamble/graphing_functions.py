import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from prepared_data import *
import plotly.express as px
from typing import Tuple, NamedTuple
import numpy as np

# plotly graphs


# ============================
def draw_candle_plotly(data, title):
    """
    ## Input shape 
    | date || open || high ||low || close
    """
    fig = go.Figure(data=[go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])
    fig.update_layout(title=title)
    fig.show()
    

# ============================
def draw_line_plotly(data, x_col, y_col, title):
    fig = go.Figure(go.Line(x=data[x_col], y=data[y_col]))
    fig.update_layout(title=title)
    fig.show()
    

# ============================
class SeabornFn(NamedTuple):
    plot_fn:str
    kwargs: dict 

class LogYGraph:
    """
    # This class add a side-by-side log-y scale to the graph
    """
    def __init__(self, fn:Tuple[SeabornFn]):
        self.fn = fn
        
    
    def draw(self, figsize=(15,5)):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
        for i in self.fn:
            getattr(sns, i.plot_fn)(**i.kwargs, ax=ax1)
        plt.yscale('log')
        for i in self.fn:
            getattr(sns, i.plot_fn)(**i.kwargs, ax=ax2)
        ax1.legend()
        ax2.legend()
        return fig, ax1, ax2
    
    
# ============================
class Risk_Return_Ratio_Report:
    """
    Generates a retport for the Risk-return ratio
    """
    def __init__(self, data, benchmark:pd.Series,risk_free=0) -> None:
        self.data = data
        self.benchmark = benchmark
        self.risk_free = risk_free
        
    def calc_benchmark_indepent(self):
        self.sharpe = self.data.apply(lambda x: sharpe_ratio(x, self.risk_free))
        self.sortino = self.data.apply(lambda x: sortino_ratio(x, self.risk_free))
        self.sharpe.name = 'Sharpe Ratio'
        self.sortino.name = 'Sortino Ratio'
    
    def calc_benchmark_depent(self):
        self.info_ratio = self.data.apply(lambda x: information_ratio(x, self.benchmark))
        self.m2 = self.data.apply(lambda x: m2_ratio(x, self.benchmark, self.risk_free))
        self.info_ratio.name = 'Information Ratio'
        self.m2.name = 'M2 Ratio'
        
        
    def output(self):
        a = pd.concat([self.sharpe, self.sortino, self.info_ratio, self.m2], axis=1).transpose()
        return a
    
    @staticmethod
    def make_pretty(styler):
        styler.format('{:.3f}')
        styler.set_caption('The higher the ratios the better')
        styler.bar(axis=1, align=0, height=50, width=60, color=['red', 'blue'], props="width: 120px; border-right: 1px solid black;")
        return styler
    
    def __call__(self):
        self.calc_benchmark_indepent()
        self.calc_benchmark_depent()
        df = self.output()
        
        return df.style.pipe(self.make_pretty)



# ============================
def draw_boxplot_percentile_comparision(data, xlabels:list):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.boxplot(data,labels=xlabels,showfliers=False,showmeans=True,patch_artist=True,
           boxprops=dict(facecolor='whitesmoke'),medianprops=dict(color='lightcoral',linewidth=2.5),meanprops=dict(markerfacecolor='teal',markeredgecolor='teal',markersize=8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis='y')
    
    return fig, ax


# ============================
def draw_ecdf(ax, data, label):
    N= len(data)
    x = np.sort(data)
    y = np.arange(N) / float(N)
    ax.plot(x, y, marker='o')
    ax.set_title(f'CDF of {label}')
    
    x_80 = np.percentile(data, 80)
    x_100 = np.percentile(data, 100)
    step_range = (x_100 - x_80)/10 
    ax.axvline(x_80, color='orange')
    ax.axvline(x_100, color='orange')
    ax.annotate(f'<- {x_100/x_80:.2f} ->', (x_80 + step_range*2, 0.4), fontsize=10)
    


# ============================
class DrawEfficientFrontier:
    def __init__(self, data, x, y, hue, hover) -> None:
        self.data = data
        self.x = x
        self.y = y
        self.hue = hue
        self.hover = hover

    def draw_efficient_frontier(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data[self.x], 
                                y=self.data[self.y], 
                                hovertext=self.data[self.hover],
                            #- Add color scale for sharpe ratio   
                            marker=dict(color=self.data[self.hue], 
                                        showscale=True, 
                                        size=7,
                                        line=dict(width=1),
                                        colorscale="RdBu",
                                        colorbar=dict(title="Sharpe<br>Ratio")
                                        ), 
                            mode='markers'))
        
        self.fig = fig
        return fig
    
    def decorate_fig(self, xlabel, ylabel):
        self.fig.update_layout(template='plotly_white',
                  xaxis=dict(title=xlabel),
                  yaxis=dict(title=ylabel),
                  title='Sample of Random Portfolios',
                  coloraxis_colorbar=dict(title="Sortino Ratio"))

    def annot_max_sortino(self):
        max_sortino = self.data[self.data[self.hue] == self.data[self.hue].max()]
        self.max_sortino = max_sortino
        self.fig.add_traces(
            px.scatter(max_sortino, x=self.x, y=self.y, hover_data=[self.hover]).update_traces(marker_size=20, marker_color="yellow").data
        )
        return self.fig
    
    def get_max_sortino(self):
        return self.max_sortino
    
    def __call__(self, xlabel, ylabel):
        self.draw_efficient_frontier()
        self.decorate_fig(xlabel, ylabel)
        return self.annot_max_sortino()
    

# ============================
class TwoVariablePortfolioReport:
    """
    # Display the ratio of two variable in a multiple assets portfolio
    """
    
    def __init__(self, data) -> None:
        self.data = data
        self.rp = {}
        
    def add_steps(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        
    def seed_weights(self, var1_weight, var2_weight):
        num_assets = len(self.data.columns)
        weights = np.zeros(num_assets)
        weights[0] = var1_weight
        weights[1] = var2_weight
        weights[2:] = ((1 - var1_weight - var2_weight) / (num_assets - 2))
        return weights
    
    def create_portfolio_stat(self):
        for i in self.var1:
            for j in self.var2:
                if i + j > 1:
                    rp = None
                else:
                    PP = PortfolioPerformance(self.data, weights=self.seed_weights(i, j))
                    rp = PP.calc_portfolio_pct()
                    rp.fillna(0, inplace=True)
                if i in self.rp:
                    self.rp[i].append((j, rp))
                else:
                    self.rp[i] = [(j, rp)]
    
    def create_volatility_rp(self, label1, label2):
        result_dict = {}
        for index, value in self.rp.items():
            renamed_index = f'{index*100:.0f}% {label1}'
            for i in value:
                if index + i[0] > 1:
                    volitality = np.nan
                else:
                    volitality = i[1]['portfolio_pct'].std()
                if renamed_index in result_dict:
                    result_dict[renamed_index].append(volitality)
                else:
                    result_dict[renamed_index] = [volitality]
                    
        df = pd.DataFrame(result_dict)
        df.index = [f'{i*100:.0f}% {label2}' for i in self.var2]
        return df.style.highlight_min(axis=None, color='gold').set_caption('Volitality report')
    
    def create_ratio_rp(self, ratio_fn, label1, label2, fn_label):
        result_dict = {}
        for index, value in self.rp.items():
            renamed_index = f'{index*100:.0f}% {label1}'
            for i in value:
                if index + i[0] > 1:
                    rp = np.nan
                else:
                    rp = ratio_fn(i[1]['portfolio_pct'])
                if renamed_index in result_dict:
                    result_dict[renamed_index].append(rp)
                else:
                    result_dict[renamed_index] = [rp]
                    
        df = pd.DataFrame(result_dict)
        df.index = [f'{i*100:.0f}% {label2}' for i in self.var2]
        return df.style.highlight_max(axis=None, color='gold').set_caption(f'{fn_label} report')
                
    def create_var_rp(self, var_fn, confidence, label1, label2, fn_label):
        result_dict = {}
        for index, value in self.rp.items():
            renamed_index = f'{index*100:.0f}% {label1}'
            for i in value:
                if index + i[0] > 1:
                    rp = np.nan
                else:
                    rp = var_fn(i[1]['portfolio_pct'], confidence)
                if renamed_index in result_dict:
                    result_dict[renamed_index].append(rp)
                else:
                    result_dict[renamed_index] = [rp]
                    
        df = pd.DataFrame(result_dict)
        df.index = [f'{i*100:.0f}% {label2}' for i in self.var2]
        return df.style.highlight_max(axis=None, color='gold').set_caption(f'{fn_label} at {confidence} percentile report')
    