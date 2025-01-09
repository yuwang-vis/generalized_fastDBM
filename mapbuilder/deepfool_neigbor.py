import torch as T
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors
# import make
from functools import cache

import numpy as np

from functools import partial

from torch.func import jacrev, vmap
# from torch.autograd.functional import jacrev, vmap
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset



### neighbor
class Neighbors:
    def __init__(
        self,
        inverter, #: nn.Module, proj.invere_transform
        classifier: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        grid_points: T.Tensor,
        inverted_grid: np.ndarray,
        labels = None,
    ) -> None:
        self.inverter = inverter
        self.classifier = classifier
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.grid = grid_points
        self.grid_width = int(np.sqrt(self.grid.shape[0]))
        self.inverted_grid = inverted_grid
        self.n_grid_points = self.grid.shape[0]
        self.global_neighbor_finder = NearestNeighbors(
            n_neighbors=10
        )  # TODO we only want 1 for now, though.
        self.global_neighbor_finder.fit(self.X_train)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.n_classes = len(np.unique(y_train))
        self.per_class_neighbor_finder = {}
        for cl in range(self.n_classes):
            neighbor_finder = NearestNeighbors(n_neighbors=5)
            neighbor_finder.fit(self.X_train[self.y_train == cl])
            self.per_class_neighbor_finder[cl] = neighbor_finder

    @cache
    def get_distance_to_nearest_neighbor(self):
        # with T.no_grad():
        #     inverted_grid = self.inverter(self.grid).cpu().numpy()
        dist, _ = self.global_neighbor_finder.kneighbors(
            self.inverted_grid, n_neighbors=1, return_distance=True
        )
        return dist.reshape((self.grid_width, self.grid_width)).copy()

    @cache
    def get_distance_to_nearest_same_class_neighbor(self):
        # this retains the data type, which might be handy.
        distances = np.zeros((self.n_grid_points,), dtype=np.float32)
        # with T.no_grad():
        #     inverted_grid = self.inverter(self.grid)
        #     inverted_grid_classes = self.classifier.classify(inverted_grid)
        #     inverted_grid = inverted_grid.cpu().numpy()
        #     inverted_grid_classes = inverted_grid_classes.cpu().numpy()
        # inverted_grid_classes = self.y_train
        ## or
        inverted_grid_classes = self.classifier.classify(T.tensor(self.inverted_grid).to(self.device)).cpu().numpy()

        for cl in range(self.n_classes):
            mask = inverted_grid_classes == cl
            # print(sum(mask))
            if sum(mask) == 0:
                print(f'no pixels of the class {cl} !!!!!!!!!!!!!!!!!')
                continue
            dist, _ = self.per_class_neighbor_finder[cl].kneighbors(
                self.inverted_grid[inverted_grid_classes == cl],
                n_neighbors=1,
                return_distance=True,
            )
            # dist, _ = self.per_class_neighbor_finder[cl].kneighbors(
            #     self.X_train[self.y_train == cl],
            #     n_neighbors=1,
            #     return_distance=True,
            # )
            dist = dist.squeeze()
            distances[mask] = dist

        return distances.reshape((self.grid_width, self.grid_width)).copy()

    @cache
    def get_distance_to_nearest_diff_class_neighbor(self):
        distances = np.full((self.n_grid_points,), np.inf, dtype=np.float32)
        # with T.no_grad():
        #     inverted_grid = self.inverter(self.grid)
        #     inverted_grid_classes = self.classifier.classify(inverted_grid)

        #     inverted_grid = inverted_grid.cpu().numpy()
        #     inverted_grid_classes = inverted_grid_classes.cpu().numpy()
        inverted_grid_classes = self.classifier.classify(T.tensor(self.inverted_grid).to(self.device)).cpu().numpy()
        # inverted_grid_classes = self.y_train

        for cl in range(self.n_classes):
            mask = inverted_grid_classes == cl
            if sum(mask) == 0:
                print(f'no pixels of the class {cl}')
                continue
            elems = self.inverted_grid[inverted_grid_classes == cl]
            for other_cl in range(self.n_classes):
                if cl == other_cl:
                    continue

                dist, _ = self.per_class_neighbor_finder[other_cl].kneighbors(
                    elems, n_neighbors=1, return_distance=True
                )
                dist = dist.squeeze()
                distances[mask] = np.minimum(distances[mask], dist)
        return distances.reshape((self.grid_width, self.grid_width)).copy()


### deepfool
def _model_activations(model, inputs):
    acts = model.activations(inputs)
    return acts, acts


# Looks ugly because of all the [...[None]][0] workarounds to work with vmap().
# It's fast though.
def _calc_r_i(output, jacobian, orig_class):
    grad_diffs = jacobian - jacobian[orig_class[None]][0]
    output_diffs = output - output[orig_class[None]][0].unsqueeze(-1)
    perturbations = T.abs(output_diffs) / T.norm(grad_diffs, dim=1)
    l_hat = T.argsort(perturbations)[0]

    r_i = (
        (perturbations[l_hat[None]][0] + 1e-4)
        * grad_diffs[l_hat[None]][0]
        / T.norm(grad_diffs[l_hat[None]][0])
    )
    return r_i


def deepfool_batch(
    model: nn.Module, input_batch: T.Tensor, labels:np.ndarray|None, max_iter: int = 50, overshoot: float = 0.02
):
    if labels is None:
        with T.no_grad():
            _, orig_classes = T.max(model(input_batch), dim=1)
            orig_classes = orig_classes.flatten()
    else:
        orig_classes = T.tensor(labels, device=input_batch.device, dtype=T.long)
    perturbed_points = input_batch.clone().detach()
    r_hat = T.zeros_like(perturbed_points)
    perturbed_points_final = T.full_like(perturbed_points, T.nan)
    perturbed_classes = orig_classes.clone().detach()
    q = T.arange(input_batch.size(0), device=perturbed_points.device, dtype=T.long)
    loop = range(max_iter)
    for i in loop:
        # loop.set_description(f"{len(q) = }")
        if len(q) == 0:
            break
        jacobians, outputs = vmap(
            jacrev(partial(_model_activations, model), has_aux=True)
        )(perturbed_points[q])
        r_is = vmap(_calc_r_i)(outputs, jacobians, orig_classes[q])
        r_hat[q] += r_is
        perturbed_points[q] = input_batch[q] + ((1 + overshoot) * r_hat[q])
        _, new_classes = T.max(model(perturbed_points[q]), dim=1)

        (changed_classes,) = T.where(new_classes != orig_classes[q])
        perturbed_classes[q[changed_classes]] = new_classes[changed_classes]
        perturbed_points_final[q[changed_classes]] = perturbed_points[
            q[changed_classes]
        ]

        q = q[T.where(new_classes == orig_classes[q])]

    mask = T.isnan(perturbed_points_final)
    perturbed_points_final[mask] = perturbed_points[mask]
    perturbed_classes[mask.any(dim=1)] = orig_classes[mask.any(dim=1)]
    return perturbed_points_final, orig_classes, perturbed_classes


def deepfool(model: nn.Module, input_example: T.Tensor, max_iter: int = 50):
    return deepfool_batch(model, input_example[None, ...], max_iter=max_iter)


def deepfool_minibatches(
    model: nn.Module,
    input_batch: T.Tensor,
    labels: np.ndarray | None,
    batch_size: int = 1000,
    max_iter: int = 50,
):  
    

    if labels is None:
        minibatches = DataLoader(
            TensorDataset(input_batch), batch_size=batch_size, shuffle=False
        )

        all_perturbed_points = []
        all_orig_classes = []
        all_perturbed_classes = []
        for batch in minibatches:
            (to_perturb,) = batch
            perturbed, orig_classes, perturbed_classes = deepfool_batch(
                model, to_perturb, labels, max_iter=max_iter
            )

            all_perturbed_points.append(perturbed)
            all_orig_classes.append(orig_classes)
            all_perturbed_classes.append(perturbed_classes)

        return (
            T.cat(all_perturbed_points, dim=0),
            T.cat(all_orig_classes, dim=0),
            T.cat(all_perturbed_classes, dim=0),
        )

    else:
        minibatches = DataLoader(
            TensorDataset(input_batch, T.tensor(labels, device=input_batch.device)),
            batch_size=batch_size,
            shuffle=False,
        )
        all_perturbed_points = []
        all_perturbed_classes = []
        for batch in minibatches:
            to_perturb, label_batch = batch
            perturbed, orig_classes, perturbed_classes = deepfool_batch(
                model, to_perturb, label_batch, max_iter=max_iter
            )

            all_perturbed_points.append(perturbed)
            all_perturbed_classes.append(perturbed_classes)

        return (
            T.cat(all_perturbed_points, dim=0),
            labels,
            T.cat(all_perturbed_classes, dim=0),
        )


if __name__ == "__main__":
    # from map_evaluation import P_wrapper, Evaluator
    import os
    import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # import matplotlib.colors as colors
    from sklearn.model_selection import train_test_split
    # blob
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import PIL
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


    data_dir = '../sdbm/data'
    data_dirs = [
        'blobs_dim100_n1000'
        # 'har', 
        # 'mnist', 
        # 'fashionmnist', 
        # 'reuters', 
        ]
    datasets_real = {}

    for d in data_dirs:
        dataset_name = d

        # X = np.load(os.path.join(data_dir, d,  'X.npy'))
        # y = np.load(os.path.join(data_dir, d, 'y.npy'))

        #blobs
        X, y = make_blobs(n_samples=1000, n_features=100, centers=5, cluster_std=1, random_state=42)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        ######
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        train_size = min(int(n_samples*0.9), 500)
        test_size = 5000 # inverse
        
        dataset =\
            train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
        datasets_real[dataset_name] = dataset
        # print(X.shape)

        ## clip dataset[1] and dataset[3] to test_size if they are larger
        if dataset[1].shape[0] > test_size:
            dataset[1] = dataset[1][:test_size]
            dataset[3] = dataset[3][:test_size]
            
    # classifiers = {
        # 'SVM': SVC(probability=True),
        # 'Random Forests': RandomForestClassifier(n_jobs=-1, random_state=0), ##random_state=42
        # 'Neural Network': MLPClassifier([200,200,200], random_state=42) ,  
        # 'SVM': cuSVC(probability=True),
        # 'Logistic Regression': linear_model.LogisticRegression(n_jobs=-1),
        ## random_state=42
    # }

    # P_ssnp = P_wrapper(ssnp=ssnp)
    projectors = {
            
                'DeepView_0.65': P_wrapper(deepview=1),
                'DBM_orig_keras': P_wrapper(NNinv_Keras=1),
                 'SSNP' : P_wrapper(ssnp=1),
                # 'DBM_orig_torch': P_wrapper(NNinv_Torch=1),
                
                }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    for data_name, dataset in datasets_real.items():
        X_train, X_test, y_train, y_test = dataset
        n_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]

        X_train, X_test, y_train, y_test = map(
                    partial(T.tensor, device=device), (X_train, X_test, y_train, y_test)
                )
        print(type(X_train))
        # X_train = T.tensor(X_train, device=device)
        # X_test = T.tensor(X_test, device=device)
        # train_dl = DataLoader(
        #     TensorDataset(X_train, y_train),
        #     batch_size=128,
        #     shuffle=True,
            # )
        print(X_train.shape) 
        print(y_train.shape)
        clf = LogisticRegression(input_dim, n_classes).to(device)
        clf.init_parameters()
        clf.fit(TensorDataset(X_train, y_train), epochs=150)
        y_test_pred = clf.predict(X_test.cpu())
        print(f"Accuracy: {accuracy_score(y_test.cpu().numpy(), y_test_pred)}")

        ################3
        import datetime 
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        ### PLEASE CHECK THE SAVE NAME BEFORE RUNNING!!!!!!!!
        save_name = 'eval_real_BLOB_CS_fixwidth' + date + '.csv'
        # load_data =  './eval_results/eval_real_BLOB_CS2023-04-20_12:16:35.csv'
        load_data = False
        data_to_evaluate = datasets_real
        data_path = 'calcu_results_BLOB_CS_fixwidth_500'
        classifiers = {'Logistic Regression': clf}
        evaluater = Evaluator()
        res = evaluater.evaluate_all(classifiers, projectors, data_to_evaluate, read_from_file=load_data, save_name=save_name, save_path=data_path)
        ##########


        for proj_name, proj in projectors.items():
            print(f"processing {data_name} with {proj_name}")
  

            proj.fit(X_train.cpu().numpy(), y_train.cpu().numpy(), clf)
            if proj.ssnp:
                X_train_2d = proj.transform(X_train.cpu().numpy())
            else:
                X_train_2d = proj.P.embedding_

            x_min, x_max = X_train_2d[:, 0].min(), X_train_2d[:, 0].max()
            y_min, y_max = X_train_2d[:, 1].min(), X_train_2d[:, 1].max()
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
            )   
            XY = np.c_[xx.ravel(), yy.ravel()]
            pixel_nd = proj.inverse_transform(XY)
            pixel_labels = clf.predict(pixel_nd) 
            # savve nerest neighbor distance map
            if not os.path.exists(f"./distance"):
                os.mkdir(f"./distance")
            np.save(f'./distance/{data_name}_{proj_name}_labels.npy', pixel_labels)

            pixel_labels = pixel_labels.reshape(xx.shape) / (n_classes - 1)
            pixel_labels = cm.tab10(pixel_labels)
            pixel_labels = np.flip(pixel_labels, axis=0)

            print(pixel_labels.shape)
            # save the pixel as image
            if not os.path.exists(f"./distance/fig"):
                os.mkdir(f"./distance/fig")
            plt.imsave(f'./distance//fig/{data_name}_{proj_name}_pixel.png', pixel_labels)

            neigbor_finder = Neighbors(proj.inverse_transform, clf, 
                    X_train.cpu().numpy(), y_train.cpu().numpy(), XY, pixel_nd)
            any_neigb = neigbor_finder.get_distance_to_nearest_neighbor()
            print(any_neigb.shape)
            # same_neigb = neigbor_finder.get_distance_to_nearest_same_class_neighbor()
            # print(same_neigb.shape)
            # diff_neigb = neigbor_finder.get_distance_to_nearest_diff_class_neighbor()
            # print(diff_neigb.shape)

            with T.no_grad():
                inverted_grid = T.tensor(pixel_nd, device=device, dtype=T.float32)
                (
                    perturbed,
                    orig_class,
                    naive_wormhole_data,
                ) = (
                    deepfool_batch(clf, inverted_grid)
                    if inverted_grid.is_cuda
                    else deepfool_minibatches(clf, inverted_grid, batch_size=1000)
                )

                dist_map = (
                    T.linalg.norm(inverted_grid - perturbed, dim=-1).cpu().numpy()
                )
                perturbed = perturbed.cpu().numpy()
                naive_wormhole_data = naive_wormhole_data.cpu().numpy()

            ## save the distance map as image
            ## make sure the directory exists
            print(dist_map.max(), 'max!!!~~~~~')
            print(any_neigb.max(), 'max!!!~~~~~')
            # print(same_neigb.max(), 'max!!!~~~~~')
            # print(diff_neigb.max(), 'max!!!~~~~~')

            #save 2d data
            np.save(f'./distance/{data_name}_{proj_name}_2d.npy', X_train_2d)
            np.save(f'./distance/{data_name}_{proj_name}_dist_map.npy', dist_map)
            np.save(f'./distance/{data_name}_{proj_name}_any_neigb.npy', any_neigb)
            # np.save(f'./distance/{data_name}_{proj_name}_same_neigb.npy', same_neigb)
            # np.save(f'./distance/{data_name}_{proj_name}_diff_neigb.npy', diff_neigb)
            # np.save(f'./distance/{data_name}_{proj_name}_perturbed_class.npy', naive_wormhole_data)
            
            # naive_wormhole_data = np.flip(naive_wormhole_data.reshape(200, 200), axis=0) / (n_classes - 1)
            # naive_wormhole_data = cm.tab10(naive_wormhole_data)            
            # plt.imsave(f"./distance/fig/{data_name}_{proj_name}_perturbed_class.png", naive_wormhole_data)

            #flip
            dist_map = np.flip(dist_map.reshape(200, 200), axis=0)
            ## save the distance map as image via plt with cmap 'jet'
            plt.imsave(f"./distance/fig/{data_name}_{proj_name}_dist_map.png", dist_map) #, cmap='jet')

            any_neigb = np.flip(any_neigb.reshape(200, 200), axis=0)
            plt.imsave(f"./distance/fig/{data_name}_{proj_name}_any_neigb.png", any_neigb) #, cmap='jet')
            # PIL.Image.fromarray(
            #     (any_neigb.reshape(200, 200) * 255).astype(np.uint8)
            # ).save(f"./distance/fig/{data_name}_{proj_name}_any_neigb.png")

            # same_neigb = np.flip(same_neigb.reshape(200, 200), axis=0)
            # # PIL.Image.fromarray(
            # #     (same_neigb.reshape(200, 200) * 255).astype(np.uint8)
            # # ).save(f"./distance/fig/{data_name}_{proj_name}_same_neigb.png")
            # plt.imsave(f"./distance/fig/{data_name}_{proj_name}_same_neigb.png", same_neigb) #, cmap='jet')

            # diff_neigb = np.flip(diff_neigb.reshape(200, 200), axis=0)
            # # PIL.Image.fromarray(
            # #     (diff_neigb.reshape(200, 200) * 255).astype(np.uint8)
            # # ).save(f"./distance/fig/{data_name}_{proj_name}_diff_neigb.png")
            # plt.imsave(f"./distance/fig/{data_name}_{proj_name}_diff_neigb.png", diff_neigb) #, cmap='jet')


            # plt.imshow(dist_map.reshape(200, 200))
            # plt.show()