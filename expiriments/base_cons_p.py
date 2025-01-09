import sys
sys.path.append(".")

from mapbuilder import MapBuilder
from invprojection import  RBFinv, Pinv_ilamp, NNinv_torch, SSNP, PPinvWrapper

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

## import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, minmax_scale
# import minmaxscale
# from sklearn.preprocessing import minmaxscale
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.datasets import load_iris

import datetime 
date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
import time
import pandas as pd
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), 
              "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

print('Start')



PPinv_dict = {
    'DBM (UMAP+NNInv)': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=NNinv_torch()),
    "SSNP": SSNP(verbose=0)
}

classifiers = {
    'Logistic Regression': LogisticRegression(n_jobs=-1),
    'Random Forests': RandomForestClassifier(n_jobs=-1, random_state=999), ##random_state=42
    'Neural Network': MLPClassifier([1024,1024,1024], random_state=999) ,  
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(),
}


data_dir = '../sdbm/data'
data_dirs = [
    # 'Iris',
    'MNIST',
    'HAR', 
    'FashionMNIST', 
    #  'reuters', 
     ]
datasets_real = {}

for d in data_dirs:

    dataset_name = d
    if d == 'Iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
        X = minmax_scale(X)

    else:
        print(f"Loading {d}")
        X = np.load(os.path.join(data_dir, d.lower(),  'X.npy'))
        y = np.load(os.path.join(data_dir, d.lower(), 'y.npy'))
        
        # n_classes = len(np.unique(y))
    n_samples = X.shape[0]
    train_size = min(int(n_samples*0.8), 5000)
    test_size = 5000 # inverse

    dataset =\
        train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
    datasets_real[dataset_name] = dataset

    ## clip dataset[1] and dataset[3] to test_size if they are larger
    if dataset[1].shape[0] > test_size:
        dataset[1] = dataset[1][:test_size]
        dataset[3] = dataset[3][:test_size]

    



if __name__ == '__main__':
    GRID = 256

    results = pd.DataFrame(columns=['Data', 'PPinv', 'Classifier', 'Accuracy', 'time dummy', 'time fast', 'grid', '$Cons_p$', '$Cons_p$ fast', 'diff $Cons_p$'])

    ### save directory
    save_dir = f'./results/cons_p/'
    os.makedirs(save_dir, exist_ok=True)

    for data_name, dataset in datasets_real.items():
        print(f"Data: {data_name}")
        X, _, y, _ = dataset
        for ppinv_name, ppinv in PPinv_dict.items():
            print(f"PP: {ppinv_name}")
            print('X shape:', X.shape)
            print('y shape:', y.shape)
            ppinv.fit(X=X, y=y)
            try:
                X2d = ppinv.X2d
            except:
                X2d = ppinv.transform(X)

            for clf_name, clf in classifiers.items():
                print(f"Classifier: {clf_name}")
                clf.fit(X, y)
                acc = clf.score(X, y)
                print(f"Accuracy: {acc}")
                print("")
                mapbuilder = MapBuilder(ppinv=ppinv, clf=clf, X=X, y=y, scaling=0.9, X2d=X2d)

                time0 = time.time()
                label_gt,_, _ = mapbuilder.get_map(content='label', resolution=GRID, fast_strategy=False)
                label_round_gt,_, _ = mapbuilder.get_map(content='label_roundtrip', resolution=GRID, fast_strategy=False)
                cons_p_gt = np.sum(label_gt != label_round_gt) / GRID**2
                time1 = time.time()
                time_diff = time1 - time0

                time2 = time.time()
                label_fast,_, _ = mapbuilder.get_map(content='label', resolution=GRID, fast_strategy=True) 
                label_round_fast,_, _ = mapbuilder.get_map(content='label_roundtrip', resolution=GRID, fast_strategy=True)
                cons_p_fast = np.sum(label_fast != label_round_fast) / GRID**2
                time3 = time.time()
                time_diff_fast = time3 - time2

                diff_cons_p = cons_p_fast - cons_p_gt

                print(f"Cons_p: {cons_p_gt}", f"Cons_p fast: {cons_p_fast}", f"Diff Cons_p: {diff_cons_p}", f'Time dummy: {time_diff}', f'Time fast: {time_diff_fast}')


                results = results._append({'Data': data_name, 'PPinv': ppinv_name, 'Classifier': clf_name, 'Accuracy': acc, 'time dummy': time_diff, 'time fast': time_diff_fast, 'grid': GRID, '$Cons_p$': cons_p_gt, '$Cons_p$ fast': cons_p_fast, 'diff $Cons_p$': diff_cons_p}, ignore_index=True)
                
                results.to_csv(f'{save_dir}/results{date}.csv')

                


            