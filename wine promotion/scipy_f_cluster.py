from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from bases import BaseClass
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ScipyFcluster(metaclass=BaseClass):
    def __init__(self, data, normalize=True, method='complete'):
        """
        Create Z-linkage distance matrix from data params
        data must be an aggregated table that has unique index label
        data is normalized by sklearn.StdScaler
        """
        self.data = data
        self.normalize = normalize
        self.Z = linkage(data, method=method)
    
    def normalize_data(self):
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        scaled_df.dropna(inplace=True)
        self.data = scaled_df
        
    def __post_init__(self):
        if self.normalize:
            self.normalize_data()
        
    
    def draw_dendrogram(self, figsize=(20, 20), **kargs):
        fig, ax = plt.subplots(figsize=figsize)
        dn = dendrogram(self.Z, labels=self.data.index, **kargs)
        self.dn = dn
        plt.show()
        
    def get_clustered(self):
        index_list = zip(self.dn['ivl'], self.dn['leaves_color_list'])
        self.clustered_data = list(index_list)
        return list(index_list)
    
    def get_clustered_by_label(self, label):
        re = filter(lambda x: x[1] == label, self.clustered_data)
        return list(re)
        
    
    
        
        
    