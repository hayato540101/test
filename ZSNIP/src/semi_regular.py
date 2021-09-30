




plt.style.use('ggplot') # これだとダークテーマでも目盛りが見える!
plt.rcParams["figure.figsize"]=[12, 8]



########################################################
def set_seed()-> None:
    import os
    import random
    import numpy as np
    import torch

    SEED_VALUE = 1234  # これはなんでも良い
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)  # PyTorchを使う場合


    '''
    ただし、PyTorchでGPUを使用する場合はさらに、
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    を設定しないと、GPU利用時には再現性を担保できません。

    ですが、torch.backends.cudnn.deterministic = Trueは、GPUでの計算速度を下げてしまいます。
    そのため私の場合、実行速度を優先し、GPUでの学習の再現性の担保までは求めないです。
    '''
########################################################