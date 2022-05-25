import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff

def plot_catplot_columns(df, cols, plot_func, grid_cols=3, figsize=(8, 6), grid=False, **kwargs):
    """
    Hàm này nhận 1 df và list of columns cần đc plot
    Tạo 1 grid và plot dữ liệu theo plot_func
    plot_func must be catplot (only accepts x column)
    """
    num_rows = round(len(cols)/grid_cols)
    fig, axes = plt.subplots(num_rows, grid_cols, figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for col, ax in zip(cols, axes.ravel()): # gán mỗi column trong cols_list to an ax-canvas
        plot_func(col, data=df, ax=ax, **kwargs)
        plt.xlabel(col)
        if grid==False:
            ax.grid(None)
        else:
            ax.grid(which="both", axis="y")
    fig.delaxes(axes[1,2])
    
    
def plot_lineplot_comparison(df_1, df_2, x, cols, legend, grid_cols=3, x_step=3, figsize=(8, 6)):
    """
    Hàm này nhận vào 2 df và 1 list of columns
    Tạo 1 grid và plot lineplot 2 column side-by-side 
    """
    
    num_rows = round(len(cols)/grid_cols)
    fig, axes = plt.subplots(num_rows, grid_cols, figsize=figsize)
    fig.suptitle("Comparison plots")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for col, ax in zip(cols, axes.ravel()): # gán mỗi column trong cols_list to an ax-canvas
        sns.lineplot(x=x, y=col, data=df_1, ax=ax, ci=None)
        sns.lineplot(x=x, y=col, data=df_2, ax=ax, linestyle='--', ci=None)
        ax.set(xlabel = "", ylabel="", title = col, 
               xticks = np.arange(df_1[x].min(), df_1[x].max()+1, x_step),  # dùng df.max + 1 vì np.arrange là [start, stop)
               )
        ax.grid(axis="x")
    fig.legend(legend)
    fig.delaxes(axes[1,2])
    
    
def draw_plotly_f_cluster(df, ticks, orient='left', color_threshold=15):
    """
    Nhận 1 pivot table. Có
    - pd.pivot(index=['target_of_cluster, e.g city, region ...'], values=['target_column'], columns=['hue, e.g year, sex ....'])
    - Ticks == list(df.index)
    MUST CALL fig.show() to show the image
    """
    fig = ff.create_dendrogram(df,labels=ticks,orientation=orient,color_threshold=color_threshold)
    fig.update_layout(width=800, height=1500)
    return fig
        
    
