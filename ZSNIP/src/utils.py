import logging
import sys, os
import config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# import lightgbm as lgb
from lightgbm import LGBMClassifier


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    # LOGGERはログの出力を命令するオブジェクト
    LOGGER = logging.getLogger()
    # 右の表示を無効にするコード2021-07-01 10:01:11,738 - DEBUG - findfont: score(<Font 'STIXNonUnicode' (STIXNonUniIta.ttf) italic normal 400 normal>) = 11.05
    logging.getLogger('matplotlib.font_manager').disabled = True
    # ログの出力形式を設定
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        # どうやってログ出力するか操作するオブジェクト　
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        # ログレベルINFO
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")

    if not os.path.isdir('./logs'):
        os.makedirs('./logs')

    return LOGGER


def newf(newfuture,logger):
    logger.info('Add new feature=> {0}'.format(newfuture))

class Feature():    
    def __init__(self):
        self.train_path = config.TRAIN_FEATTURE_SAVE_PATH
        self.test_path = config.TEST_FEATTURE_SAVE_PATH

    def load(self):
        adfetr = pd.DataFrame(index=[])
        for i in os.listdir(config.TRAIN_FEATTURE_SAVE_PATH):
            # print(os.path.join(config.TRAIN_FEATTURE_SAVE_PATH,i))
            col = pd.read_pickle(os.path.join(config.TRAIN_FEATTURE_SAVE_PATH,i))
            adfetr = pd.concat([adfetr,col],axis='columns')
            
        adfete = pd.DataFrame(index=[])
        for i in os.listdir(config.TEST_FEATTURE_SAVE_PATH):
            # print(os.path.join(config.TEST_FEATTURE_SAVE_PATH,i))
            col = pd.read_pickle(os.path.join(config.TEST_FEATTURE_SAVE_PATH,i))
            adfete = pd.concat([adfete,col],axis='columns')
        print('additional_tr.shape:{},additional_te.shape:{}'.format(adfetr.shape, adfete.shape))
        display(adfetr, adfete)
        return adfetr, adfete 



    def save_tr(self,df,cols):
        '''
        準備：特徴量を作る

        保存方法
        作った特徴量の列一覧をdf_train.columns[-17:]のように後ろスライスでとってきて取得、colsにコピペしてsaveを実行することで保存する。
        挙動は上書きのはずなので注意  

        読みこみ方法(pathにバックスラッシュを使わないこと)
        a = pd.read_pickle('../feature/train/ID_TO_BIRTH_RATIO_train.pkl')
        b = pd.concat([df_train,a],axis=1) # axis=1
        '''
        for col in cols:
            print(self.train_path + str(col)+'_train.pkl')
            df[col].to_pickle(self.train_path + str(col)+'_train.pkl')
    
    def save_te(self,df,cols):
        for col in cols:
            print(self.test_path + str(col)+'_test.pkl')
            df[col].to_pickle(self.test_path + str(col)+'_test.pkl')


class ModelFactory(object):
    def __init__(self, name, config, logger):
        logger.info('Selecting model => {0}'.format(name))
        if name == 'LogisticRegression':
            self.model = LogisticRegression(**config[name])
        elif name == 'RandomForest':
            self.model = RandomForestClassifier()
        elif name == 'LightGBM':
            self.model = LGBMClassifier()
        else:
            logger.error('{0} is not implemented'.format(name))
            raise NotImplementedError()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        prediction = self.model.predict_proba(X)
        return prediction

    def predict_class(self, X):
        prediction = self.model.predict(X)
        return prediction



def model_perfo(logger,auc_score):
    
    logger.info('auc_score => {0}'.format(auc_score))

def auc_plot_log(y_test, pred,logger):
    # pred = model.predict_proba(X_test)[: , 1]
    #評価用モジュール
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
    #ROC曲線プロット
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    auc_score = roc_auc_score(y_test, pred)
    plt.plot(fpr, tpr, label='AUC = %.3f' % (auc_score))
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    logger.info('auc_score => {0}'.format(auc_score))