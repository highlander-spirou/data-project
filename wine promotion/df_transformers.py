import pandas as pd
from functools import reduce
from typing import Optional, List


def create_df_for_hier_cluster(data:pd.DataFrame, index_label:str, values_label:list):
    returned_df = pd.pivot_table(data,index=[index_label],values=values_label,fill_value=0)
    return returned_df
        
def df_by_year(df, year:int, index:Optional[str]=None):
    returned_df = df.copy(deep=True)
    returned_df = returned_df.query("year == @year")
    if index is not None:
        returned_df.set_index(index, inplace=True)
    returned_df.columns.tolist()
    new_column_list = {}
    for i in returned_df:
        new_column_list[i] = i + '_' + str(year)
    returned_df.rename(columns=new_column_list, inplace=True)
    return returned_df

def merge_multiple_df(list_df:List[pd.DataFrame], on:str, index:Optional[str]=None):
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=[on], how='inner'), list_df)
    if index is not None:
        df_merged.set_index(index, inplace=True)
    return df_merged
    