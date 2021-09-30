'''
import文より前に記載しないと__doc__として認識されないらしい

このdocstiringはGoogleスタイルで記述されています。

メソッド一覧(先頭にアンスコつけるとimport *したときに読み込まれない)
- scp_heat
- miss_check
- reduce_mem_usage

Todo:
    TODOリストを記載
    * 特になし。思いついたやつ、見かけたやつでedaにあたり、使いまわせそうなメソッドを記載するだけ
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap

def scp_heat(df,cols):
    
    scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                    names=cols,alpha=0.3)
    plt.tight_layout()
    plt.show()
    import numpy as np

    cm = np.corrcoef(df[cols].values.T)

    hm = heatmap(cm, row_names=cols, column_names=cols)
    
    plt.show()
    # ex. scp_heat(df,cols)

# Function to calculate missing values by column
def miss_check(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


def reduce_mem_usage(df, verbose=True):
    """[summary]

    Args:
        df ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # 使うときは下記2行が実行ファイル上にないと上手くprintされないので注意
    # import warnings
    # warnings.simplefilter('ignore')
    print('start size(BEFORE): {:5.2f} Mb'.format(df.memory_usage().sum() / 1024**2))
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb (AFTER:{:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df









# warehouse
# Oを含む列を全取得
# temp_col = [item for item in df.columns if item.find('O') != -1]
# temp_col

# 統計量の比較？役立つかわからん
# a = train.query('Attrition == 1').describe()
# b = train.query('Attrition == 0').describe()
# a-b