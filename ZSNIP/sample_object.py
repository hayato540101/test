import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_profiling as pdp
import numpy as np

df = pd.DataFrame({'A': [[0, 1, 2], 'foo', [], [3, 4]],
                   'B': 1,
                   'C': [['a', 'b', 'c'], np.nan, [], ['d', 'e']]})

# sample code
# ある列に少数の値が入っているか確認
engagement_df[engagement_df['lp_id']%1 != 0][:30]
