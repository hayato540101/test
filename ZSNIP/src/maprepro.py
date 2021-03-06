import os
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import config
from utils import setup_logger

def application_preprocessing(df, logger, save_path):
    logger.info('Start Prerpocessing')

    threshold = round(len(df) * config.PERCENT)
    null_cnt = df[[col for col in df.columns if col not in config.unused]].isnull().sum()
    reject_col = null_cnt[null_cnt > threshold].index
    df = df[[col for col in df.columns if col not in reject_col]]

    num_col = [col for col in df.columns if df[col].dtype != 'object' and col not in config.unused]
    not_num_col = [col for col in df.columns if col not in num_col and col not in config.unused]

    df = pd.get_dummies(df, columns=not_num_col, dummy_na=True)

    logger.info('Handling Missing Values')
    for col in df.columns:
        if col in num_col:
            df[col].fillna(df[col].mean(), inplace=True)
            if config.SCALING:
                sc = StandardScaler()
                df[col] = sc.fit_transform(df[col].values.reshape(-1, 1))

    logger.info('application shape:{0}'.format(df.shape))
    logger.info('Save data to directory {0}'.format(save_path))

    train = df[~df['TARGET'].isnull()]
    test = df[df['TARGET'].isnull()].drop(['TARGET'], axis=1)
    train.to_pickle(os.path.join(save_path, 'application_train.pickle'))
    test.to_pickle(os.path.join(save_path, 'application_test.pickle'))

    logger.info('Finish Preprocessing')

from pickle import load
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
def dfpres2(df, logger):
    logger.info('Start Prerpocessing')
    
    _id = df['id'].copy()
    bfshape = df.shape

    threshold = round(len(df) * config.PERCENT)
    null_cnt = df[[col for col in df.columns if col not in config.unused]].isnull().sum()
    reject_col = null_cnt[null_cnt > threshold].index
    df = df[[col for col in df.columns if col not in reject_col]]
    # df = df[[col for col in df.columns if col not in config.Busstop]]
    df = df[[col for col in df.columns if col not in config.unused]]

    num_col = [col for col in df.columns if df[col].dtype != 'object' and col not in config.unused]
    not_num_col = [col for col in df.columns if col not in num_col and col not in config.unused]
    # df = df[df[config.Busstop]]

    df = pd.get_dummies(df, columns=not_num_col, dummy_na=True)

    logger.info('Handling Missing Values')
    if os.path.exists('../sc/scaler_object.pkl'):
        scaler = load(open("sample.pkl", "rb"))
        for col in df.columns:
            if col in num_col:
                df[col].fillna(df[col].mean(), inplace=True)
                if config.SCALING:
                    scaler.transform(df[col].values.reshape(-1, 1))
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    else:
        for col in df.columns:
            if col in num_col:
                df[col].fillna(df[col].mean(), inplace=True)
                if config.SCALING:
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
                    dump(scaler, open("../sc/sample.pkl", "wb"))
    

    # df.drop('id',inplace=True)
    df = pd.concat([_id,df],axis='columns')
    # display(df)

    afshape = df.shape
    
    print('Before:{},After:{}'.format(bfshape,afshape))
    logger.info('Before:{},After:{}'.format(bfshape,afshape))

    # logger.info('Save data to directory {0}'.format(os.path.join(config.SAVE_PATH, '3prd_tr.pkl')))

    # df.to_pickle(os.path.join(config.SAVE_PATH, '3prd_tr.pkl'))
    df.to_pickle(os.path.join(config.SAVE_PATH, '3prd_te.pkl'))
    # # df.to_pickle(os.path.join(config.SAVE_PATH, 'prd_tr_y.pkl'))

    logger.info('Finish Preprocessing')

# ----------------------------------------------?????????????????????????????????????????????---------------------------------------------------------- #  
def labeleng(df_train,df_test):
    from sklearn.preprocessing import LabelEncoder
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in df_train:
        if df_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(df_train[col])
                # Transform both training and testing data
                df_train[col] = le.transform(df_train[col])
                df_test[col] = le.transform(df_test[col])

                # Keep track of how many columns were label encoded
                le_count += 1
    print('%d columns were label encoded.' % le_count)
    
    return df_train,df_test

    
def onehoteng(df_train,df_test):
        # one-hot encoding of categorical variables
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)

    print('Training Features shape: ', df_train.shape)
    print('Testing Features shape: ', df_test.shape)
    
    return df_train,df_test

    num_col = [col for col in df.columns if df[col].dtype != 'object']
    not_num_col = [col for col in df.columns if col not in num_col and col not in config.unused]
    


if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logger = setup_logger('./logs/preprocessing_{0}.log'.format(NOW))
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_train.csv'), nrows=None)
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_test.csv'), nrows=None)
    all_df = pd.concat([train_df, test_df])
    application_preprocessing(all_df, logger, config.SAVE_PATH)


# lightGBM????????????specialjson????????????????????????????????????????????????????????????????????????????????????
def remove_specialjson(df_train,df_test):
    '''
    re.sub?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    [] ????????????????????? ^ ????????????????????????????????????????????????????????????????????????????????????
    x??????????????????????????????????????????????????????a???z??????????????????????????????????????????A???Z???????????????0???9?????????????????????????????? _ ???
    ?????????1?????????????????????????????????????????????????????????????????????????????????????????????????????????
    rename???lambda??????????????????????????????rename????????????????????????'''

    import re
    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return df_train,df_test

'''logger?????????????????????
NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'); logger = setup_logger(out_file='./src/logs/train_{0}.log'.format(NOW))'''

# warehouse
'''
# ?????????????????????????????????????????????
df.dtypes.value_counts()
# float64    65
# int64      41
# object     16

# Number of unique classes in each object column
# ?????????????????????????????????????????????????????????
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# [] ????????????????????? ^ ????????????????????????????????????????????????????????????????????????????????????
# x??????????????????????????????????????????????????????a???z??????????????????????????????????????????A???Z???????????????0???9?????????????????????????????? _ ???
# ?????????1?????????????????????????????????????????????????????????????????????????????????????????????????????????
import re
# re.sub????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????rename???lambda??????????????????????????????rename????????????????????????
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# ????????? ab+ ??? 'a' ??? 1 ???????????? 'b' ????????????????????????????????????????????? 'a' ??????????????????????????????



# df_train=df_train.select_dtypes(exclude=['int'])
# df_train=df_train.select_dtypes(exclude=['float'])

np.set_printoptions(threshold=900)
pd.set_option('display.max_columns',10000)

# 1?????????????????????dataframe?????????
df[df.columns[df.columns != 'b']]

'''
