from bases import BaseClass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from random import random

class KMeanClustering(metaclass=BaseClass):
    def __init__(self, data, normalize=True) -> None:
        self.data = data
        self.normalize = normalize
        self.kmean_obj = None
        
    def normalize_data(self):
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        scaled_df.dropna(inplace=True)
        self.data = scaled_df
    
    def __post_init__(self):
        if self.normalize:
            self.normalize_data()
    
    def run_clustering(self, stop):
        """
        number of cluster always start from two, end with stop
        """
        cluster_num = np.arange(2, stop+1)
        self.cluster_num = cluster_num
        for num in cluster_num:
            kmeans = KMeans(n_clusters = num, random_state = 21)
            kmeans.fit(self.data)
            if self.kmean_obj is None:
                self.kmean_obj = [kmeans]
            else:
                self.kmean_obj.append(kmeans)
    
    def plot_inertia_elbow_plot(self):
        inertia_list = [i.inertia_ for i in self.kmean_obj]
        plt.plot(self.cluster_num, inertia_list, marker="o")
        plt.title("Elbow plot to measure inertia of clusters")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.grid(axis="x")
        sns.despine()
        plt.show()
    
    def get_label(self):
        df = pd.DataFrame(index = self.data.index)
        for num in self.cluster_num:
            df[f"k={num}"] = self.kmean_obj[num-2].labels_
        
        return df
        