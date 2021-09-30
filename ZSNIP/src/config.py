SEED = 2019
DATA_PATH = '/home/ubuntu/kaggle/home_credit/Input/'
SAVE_PATH = '../pickle'
TRAIN_FEATTURE_SAVE_PATH = '../feature/train/'
TEST_FEATTURE_SAVE_PATH = '../feature/test/'
MODEL_NAME = 'LightGBM'
MODEL_FILE = '{0}_model.pickle'.format(MODEL_NAME)
TEST_SIZE = 0.2
PERCENT = 0.5
SCALING = True
TARGET = ['Attrition']
unused = ['id','Over18','StandardHours','Attrition']
# Busstop= ['']

MODEL_CONFIG = {
    'LogisticRegression': {
        'C': 1.0,
        'random_state': SEED,
        'max_iter': 100,
        'penalty': 'l2',
        'n_jobs': -1,
        'solver': 'lbfgs',
        #'class_weight': {0:1, 1:2},
    },
    'RandomForest': {
        'max_depth': 8,
        'min_sample_split': 2,
        'n_estimator': 200,
        'random_state': SEED,
        'class_weight': {0:1, 1:2},
    },
    'LightGBM': {
        'max_depth': 8,
        'min_sample_split': 2,
        'n_estimator': 100,
        'random_state': SEED,
        'class_weight': {0:1, 1:2},
        'objective': 'binary',
        'metric': 'auc',
#         LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#         importance_type='split', learning_rate=0.1, max_depth=-1,
#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
#         n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
#         random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
 
    }

}