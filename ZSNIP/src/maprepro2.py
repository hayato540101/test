import os
import datetime

import numpy as np
import pandas as pd

import pickle
from pickle import load
from pickle import dump

def mk_dir():
    # JSTとUTCの差分
    DIFF_JST_FROM_UTC = 9
    NOW = (datetime.datetime.utcnow() + datetime.timedelta(hours=DIFF_JST_FROM_UTC)).strftime('%Y-%m-%d_%H-%M-%S')
    # NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    TMP_DIR = '../models/'+NOW
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    return NOW,TMP_DIR

def load_models(TMP_DIR):
    print('load開始')
    modeldict = {}
    for i, file in enumerate(os.listdir(TMP_DIR)):
        # print(f'{TMP_DIR}/{file}')
        model = pickle.load(open(f'{TMP_DIR}/{file}', 'rb'))
        modeldict[i+1]=model
    print('load終わりました')
    return modeldict

def mk_pred_dict(modeldict, test_x):
    pred_dict={}
    print('予測リストが入った辞書作成開始')
    for key, model in modeldict.items():
        pred = model.predict(test_x).reshape(-1,1)
        pred_dict[key] = pred
    print('予測リストが入った辞書作成終了')
    return pred_dict

# def wei_average(pred_dict,test_length):
#     print('複数モデルを使用した重み付き平均予測を開始します')
#     pred_wei_average = np.zeros(shape=(test_length,1))
#     tmp=0
#     for weight, pred in pred_dict.items():
#         tmp+=weight
#         print(tmp)
#         pred_wei_average += weight*pred
#         pred_wei_average /= tmp
#     print('重み付き平均pred_wei_averageがreturnされます')
#     print('重み付き平均pred_wei_averageがreturnされました')
#     return pred_wei_average

def mk_output(df,NOW,PRACTICE=True):
    OUTPUT_DIR = '../../output/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if PRACTICE:
        df.to_csv(f'{OUTPUT_DIR}PRACTICE_{NOW}.csv',index=False)
        print('PRACTICE=True')
        print(f'MADE {OUTPUT_DIR}PRACTICE_{NOW}.csv')
    else:
        df.to_csv(f'{OUTPUT_DIR}{NOW}.csv',index=False)
        print('PRACTICE=False')
        print(f'MADE {OUTPUT_DIR}{NOW}.csv')
        

    # return NOW,TMP_DIR



