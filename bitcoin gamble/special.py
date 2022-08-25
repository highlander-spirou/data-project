import pandas as pd

class BarGraphWithColumAsX_axis:
    def __init__(self, data, column_as_x_axis, column_as_hue, value_label) -> None:
        """
        # Specialise transformation for bar graph plotting
        # Use get_docstring() for more details on data transformation schema and plotting result
        """
        self.data = data
        self.column_as_x_axis = column_as_x_axis
        self.column_as_hue = column_as_hue
        self.value_label = value_label

    @staticmethod
    def print_docstring():
        a = """
    # Hàm này là 1 scpecialise transformation 
    ## dùng để vẽ index (axis 0) ở trục x, column là từng bar, và giá trị là trục y
    
    ### Input
    | Date (index) || g1 || g2 || g3
    
    ### Transformation schema
    `pd.melt(data, id_vars='Date', var_name="hue_label", value_name="<input-your-name>")`
    
    ### Output
    | index (reset) || Date (id_var) || hue_label || value_label
    
    ### Plot by sns.barplot()
    █: g1
    ▓: g2
    ░: g3
    
    |
    |  ▓      ░   █
    | █▓     ▓░   █ ░
    | █▓░   █▓░   █▓░
    ---|-----|-----|---
    2017   2018   2019
    """
        print(a)

    def __call__(self):
        new_data = self.data.copy()
        return pd.melt(new_data, id_vars=self.column_as_x_axis, 
                       var_name=self.column_as_hue, value_name=self.value_label)
        
        
        
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
from typing import Iterable, Tuple, Union, TypedDict


class BrokenAxisConfig(TypedDict):
    
    # Fig-axis kwargs
    figheight:Union[float, int]
    figwidth:Union[float, int]
    ax_hspace:float
    height_ratio: int
    
    # Legend settings
    legend_title: str
    
    # Figure titles
    suptitle: str
    x_axis_title: str
    y_axis_title: str
    x_offset: int
    y_offset: int
    
    # Xticks
    rotation:int
    
    
class BrokenAxis:
    """
    ## Keyword arguments:
    
    ### Fig-axis kwargs
    - `figheight`: Control the fig overall fig height
    - `figwidth`: Control the fig overall fig width
    - `ax_hspace`: Control the gap between ax1 and ax2
    - `height_ratio`: Control how much the ax2 larger than the ax1
    
    
    ### Legend settings
    - `legend_title`: Set the title for the legend, default: "Hue"
    
    ### Figure titles
    - `suptitle`: Set the overall title for the graph
    - `x_axis_title`: Set the main x title for the graph
    - `y_axis_title`: Set the main x title for the graph
    - `x_offset`: Set the distance of the x_axis_title from the x ticks
    - `y_offset`: Set the distance of the y_axis_title from the y ticks
    
    ### Xticks
    - `ticks`: np.arrange or list 
    - `rotation`: Set rotation for xticks 
    
    """
    def __init__(self) -> None:
        pass
    
    def add_attr(self, attr_name, value):
        setattr(self, attr_name, value)
    
    def add_kwarg_attr(self, **kwargs):
        for index, value in kwargs.items():
            self.add_attr(index, value)
    
    def create_fig_axis(self):
        fig = plt.figure()
        fig.set_figheight(self.figheight)
        fig.set_figwidth(self.figwidth)
        spec = gridspec.GridSpec(ncols=1, nrows=2, wspace=0.5, hspace=self.ax_hspace, height_ratios=[1, self.height_ratio])
        ax1 = fig.add_subplot(spec[0]) # the top
        ax2 = fig.add_subplot(spec[1]) # the bottom
        
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig
        
    def config_two_graphs(self):
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=self.ax1.transAxes, color='k', clip_on=False)
        self.ax1.plot((-d, +d), (-d*self.height_ratio, +d*self.height_ratio), **kwargs)        # top-left diagonal
        self.ax1.plot((1 - d, 1 + d), (-d*self.height_ratio, +d*self.height_ratio), **kwargs)  # top-right diagonal

        kwargs.update(transform=self.ax2.transAxes)  # switch to the bottom axes
        self.ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        self.ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
         
    def ax_decorator(self):
        """
        # Manipulate the ax behavior to make the grpah looks legit
        """
        # hide the spines between ax and ax2
        self.ax1.spines['bottom'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        self.ax2.xaxis.tick_bottom()
        self.ax1.set(xlabel=None, ylabel=None)
        self.ax2.set(xlabel=None, ylabel=None)
        
                
    def draw_sns_plot(self, sns_func:str, sns_kwargs):
        """
        ### Kwargs
        
        - sns_func: sns plotting method, i.e "barplot"
        - sns_kwargs: (x='...', y='...', hue='...', data=...)
        """
        # top graph
        g1 = getattr(sns, sns_func)(**sns_kwargs, ax=self.ax1)
        
        # bottom graph
        g2 = getattr(sns, sns_func)(**sns_kwargs, ax=self.ax2)
        
        self.g1 = g1
        self.g2 = g2
        
    def crop_ax_view(self, bot_lim, top_lim):
        """
        # This method limit the axis of the top and bottom view
        """
        self.ax1.set_ylim(top_lim[0], top_lim[1])  # outliers only
        self.ax2.set_ylim(bot_lim[0], bot_lim[1])  # most of the data
        
    def config_legend(self):
        # add legend to the top graph
        self.g1.legend(title=self.legend_title, loc=2, bbox_to_anchor = (1,1))
        # remove legend from the bottom graph
        self.g2.legend_.remove()
    
    def set_xticks(self, ticks:Iterable):
        self.ax1.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=True,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        self.ax2.set_xticklabels(ticks, rotation=self.rotation)
        
    def set_fig_titles(self):
        self.fig.supxlabel(self.x_axis_title, y=self.x_offset)    
        self.fig.supylabel(self.y_axis_title, x=self.y_offset)
        self.fig.suptitle(self.suptitle)
    
    def draw_plot(self, sns_func:str, sns_kwargs, bot_lim:Tuple[float, float], top_lim:Tuple[float, float], ticks):
        self.create_fig_axis()
        self.draw_sns_plot(sns_func, sns_kwargs)
        self.crop_ax_view(bot_lim, top_lim)
        self.config_two_graphs()
        self.config_legend()
        self.set_xticks(ticks)
        self.set_fig_titles()
        self.ax_decorator()
        plt.savefig('books_read.png', bbox_inches='tight', pad_inches=.3)
        
    @staticmethod
    def generate_kwargs() -> BrokenAxisConfig:
        """
        Create a config object with default arguments
        """
        
        return {
            'ax_hspace': 0.1,
            'figheight': 6,
            'figwidth': 6,
            'height_ratio': 2,
            'legend_title': 'Hue',
            'rotation': 0,
            'suptitle': '',
            'x_axis_title': '',
            'y_axis_title': '',
            'x_offset': 0,
            'y_offset': 0,
        }