import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_bar_volume(data, volume:str, order_list=None, **kwargs):
    """
    Expected kwargs labels:
        x_label, y_label, title
    """
    if order_list is not None:
        sns.countplot(x=data[volume], data=data, order = order_list)
    else:
        sns.countplot(x=data[volume], data=data)
    plt.title(kwargs['title']) if 'title' in kwargs else None
    plt.xlabel(kwargs['x_label']) if 'x_label' in kwargs else None
    plt.ylabel(kwargs['y_label']) if 'y_label' in kwargs else None
    plt.show()
    

def plot_grid_bar_day_of_week(data, volume, col:str, order_list=None, **kwargs):
    """
    Expected kwargs labels:
        x_label, y_label, title
    """
    g = sns.catplot(x=volume, data=data, kind="count", col=col, order=order_list,sharey=False)
    g.fig.subplots_adjust(top=0.85)
    plt.suptitle(kwargs['title']) if 'title' in kwargs else None
    plt.xlabel(kwargs['x_label']) if 'x_label' in kwargs else None
    plt.ylabel(kwargs['y_label']) if 'y_label' in kwargs else None
    plt.show()
    
    
def plot_heatmap_calender(data, x_column, y_column, figsize = (30,10), title=None, **heatmap_kwargs):
    """
    - x_column will hold each hour of each day of week
    - y_column will hold the day
    """
    cross_data = pd.crosstab(data[x_column], data[y_column])
    fig, ax = plt.subplots(figsize = figsize)
    sns.heatmap(cross_data, ax = ax, fmt='g', **heatmap_kwargs)
    plt.xticks(rotation=0)
    plt.title(title)
    plt.show()

def plot_features_grid(data, cols, grid_cols=3, figsize=(20, 20), hue=None):
    num_rows = round(len(cols)/grid_cols)
    fig, axes = plt.subplots(num_rows, grid_cols, figsize=figsize, sharey=False)
    fig.tight_layout()
    for col, ax in zip(cols, axes.ravel()):
        sns.countplot(data=data, x=col, ax=ax, hue=hue)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


def plot_percent_features_grid(data, cols, grid_cols=3, figsize=(20, 20), hue=None):
    num_rows = round(len(cols)/grid_cols)
    fig, axes = plt.subplots(num_rows, grid_cols, figsize=figsize)
    fig.tight_layout()
    for col, ax in zip(cols, axes.ravel()):
        table_perc = data.groupby('serious')[col].value_counts(normalize = True, sort = False) * 100
        table_perc = table_perc.reset_index(name="percentage")
        sns.barplot(data=table_perc, x=col, y='percentage', ax=ax, hue=hue)
        labels = [i for i in range(0,105,5)]
        plt.yticks(labels)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()
    

def plot_on_map(data, hue=None, size=None):
    """
    lat and long must named "longitude" and "latitude" in dataframe
    """
    fig = px.scatter_mapbox(data, lat = 'latitude', lon = 'longitude', color = hue,
                        size = size, color_continuous_scale = px.colors.sequential.Bluered,
                        size_max = 15, zoom = 5, height = 800)
    fig.update_layout(mapbox_style = 'open-street-map')
    fig.show()