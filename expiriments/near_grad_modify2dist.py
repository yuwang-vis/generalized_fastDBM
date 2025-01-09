from base_cons_p import *
import torch

PPinv_dict = {
    # 'NNInv': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=NNinv_torch()),
    # 'DBM(t-SNE+NNInv)': PPinvWrapper(P=TSNE(n_components=2, random_state=0), Pinv=NNinv_torch()),
    # "SSNP": SSNP(verbose=0),
    # 'RBF': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=RBFinv()),
    'iLAMP': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=Pinv_ilamp()),
}

### save directory
save_dir = f'./results/grad_near/'
os.makedirs(save_dir, exist_ok=True)

GRID = 512

data_name = 'MNIST'
dataset = datasets_real[data_name]

for data_name, dataset in datasets_real.items(): ### THIS IS ALL THE DATASET!!!!
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

        mapbuilder = MapBuilder(ppinv=ppinv, clf=None, X=X, y=y, scaling=0.9, X2d=X2d)


        time0 = time.time()
        label_gt, dist_gt, _ = mapbuilder.get_map(content='dist_map_general', resolution=GRID, fast_strategy=False)
        time1 = time.time()
        time_gt = time1 - time0

        time2 = time.time()
        label_fast, dist_fast, sparse = mapbuilder.get_map(content='dist_map_general', resolution=GRID, fast_strategy=True, threshold=0.15)
        time3 = time.time()
        time_fast = time3 - time2 

        np.savez(f'{save_dir}/{data_name}_{ppinv_name}_dist_{GRID}.npz',  gt_map=dist_gt, fast_map=dist_fast, sparse=sparse, time_gt=time_gt, time_fast=time_fast, grid=GRID, X2d=X2d)

        # clean cuda memory
        del label_gt, dist_gt, label_fast, dist_fast, sparse
        torch.cuda.empty_cache()
