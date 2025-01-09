import os
import numpy as np
from queue import PriorityQueue
from scipy import interpolate
import dask.array as da
from sklearn.neighbors import KDTree
from numba_progress import ProgressBar
import torch.nn as nn
import torch as T
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
from tqdm import tqdm
from enum import Enum
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
from time import time

from .tools import binary_split, generate_windows, get_pixel_priority, get_tasks_with_same_priority, get_window_borders, get_pixel_priority_general
from .deepfool_neigbor import Neighbors, deepfool_minibatches, deepfool_batch
from .nnclassifier import NNClassifier, LogisticRegression_nn

class MapBuilder:
    def __init__(self, ppinv, clf=None, X2d=None, X=None, y=None, scaling=1, P=None):
        # y = np.array(y).reshape(-1, 1)
        # y = np.array(y).astype(int)

        assert X2d.shape[1] == 2
        assert X.shape[0] == X2d.shape[0]
        assert X.shape[1] > 2
        
        if clf is None:
            device = T.device("cuda" if T.cuda.is_available() else "cpu")   
            print('n_classes:', len(np.unique(y)))
            # self.clf = LogisticRegression_nn(X.shape[1], len(np.unique(y))).to(device)
            self.clf = NNClassifier(X.shape[1], len(np.unique(y)), layer_sizes=(100,)).to(device)
            self.clf.init_parameters()
            # X_tensor, y_tensor = map(partial(T.tensor, dtype=T.float32), (X, y))
            X_tensor, y_tensor = T.tensor(X, dtype=T.float32), T.tensor(y, dtype=T.long)
            X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
            self.clf.fit(TensorDataset(X_tensor, y_tensor), epochs=150)
        else:
            self.clf = clf

        self.ppinv = ppinv
        self.X = X
        self.X2d = X2d
        self.padding = (1-scaling) / 2
        self.scaler2d = MinMaxScaler(feature_range=(0+self.padding, 1-self.padding))
        self.X2d_scaled = self.scaler2d.fit_transform(self.X2d)


        self.y_label = y
        self.P = P

        
        # self.scalernd = scalernd
        # self.xx, self.yy = self.make_meshgrid(grid=grid)
        # print('calculating probability map')
        # self.map_res = self.get_prob_map()
        # self.gradient_res = None
        # self.inversed_feature_res = self.inversed_feature()


    def make_meshgrid(self, grid=300, x=np.array([0,1]), y=np.array([0,1])):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 0.0, x.max() + 0.0
        y_min, y_max = y.min() - 0.0, y.max() + 0.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid),
                            np.linspace(y_min, y_max, grid))
        return xx, yy
    
    def get_nd(self, xy):
        # time0 = time()
        unscaled = self.scaler2d.inverse_transform(xy)
        # time1 = time()
        nd = self.ppinv.inverse_transform(unscaled)
        # time2 = time()
        # self.time_acc += time1 - time0
        # self.time_acc2 += time2 - time1
        # self.time_called += 1
        return nd        
        # return self.ppinv.inverse_transform(unscaled)

    def get_gradient_map_general(self, xy, resolution):
        """
        for each pixel, the gradient is calculated by
        Dx(y) = (B(y + (w, 0)) - B(y - (w, 0)))/ 2w 
        Dy(y) = (B(y + (0, h)) - B(y - (0, h)))/ 2h 
        D(y) = squrt(‖Dx(y)‖**2 + ‖Dy(y)‖**2)
        
        where y is a point in the 2D projection space and w and h are a pixel's width and height, respectively.
        B is ppinv.inverse_transform() function.
        """
        # print("calculating gradient map")
        w = 1 / resolution ##############CHECK this
        X2d_l = xy - np.array([w, 0])
        X2d_r = xy + np.array([w, 0])
        X2d_u = xy + np.array([0, w])
        X2d_d = xy - np.array([0, w])
        Dx = (self.get_nd(X2d_r) - self.get_nd(X2d_l)) / (2 * w)
        Dy = (self.get_nd(X2d_u) - self.get_nd(X2d_d)) / (2 * w)
        gradients = np.sqrt(np.sum(Dx**2, axis=1) + np.sum(Dy**2, axis=1))  
        return gradients
    
    def get_gradient_map_reduced(self, xy, resolution):
        """
        for each pixel, the gradient is calculated by
        Dx(y) = (B(y + (w, 0)) - B(y - (w, 0)))/ 2w 
        Dy(y) = (B(y + (0, h)) - B(y - (0, h)))/ 2h 
        D(y) = squrt(‖Dx(y)‖**2 + ‖Dy(y)‖**2)
        
        where y is a point in the 2D projection space and w and h are a pixel's width and height, respectively.
        B is ppinv.inverse_transform() function.
        """
        XY_int = [(int(x*resolution), int(y*resolution)) for x, y in xy]
        tem_set = set()
        for (x, y) in XY_int:
            if (x-1, y) not in self.nd_sparse:
                tem_set.add((x-1, y))
            if (x+1, y) not in self.nd_sparse:
                tem_set.add((x+1, y))
            if (x, y-1) not in self.nd_sparse:
                tem_set.add((x, y-1))
            if (x, y+1) not in self.nd_sparse:
                tem_set.add((x, y+1))

        if len(tem_set) != 0:
            tem_set = np.array(list(tem_set)).reshape(-1, 2)
            tem_set_nd = self.get_nd(tem_set/ resolution)
            for i, (x, y) in enumerate(tem_set):
                self.nd_sparse[(x, y)] = tem_set_nd[i]

        gradients = []
        w = 1 / resolution ##############CHECK this
        for (x, y) in XY_int:
            Dx = (self.nd_sparse[(x+1, y)] - self.nd_sparse[(x-1, y)]) / (2 * w)
            Dy = (self.nd_sparse[(x, y+1)] - self.nd_sparse[(x, y-1)]) / (2 * w)
            gradients.append(np.sqrt(np.sum(Dx**2) + np.sum(Dy**2)))
        return np.array(gradients)

                
        # print("calculating gradient map")
        # w = 1 / resolution ##############CHECK this
        # X2d_l = xy - np.array([w, 0])
        # X2d_r = xy + np.array([w, 0])
        # X2d_u = xy + np.array([0, w])
        # X2d_d = xy - np.array([0, w])
        # Dx = (self.get_nd(X2d_r) - self.get_nd(X2d_l)) / (2 * w)
        # Dy = (self.get_nd(X2d_u) - self.get_nd(X2d_d)) / (2 * w)
        # gradients = np.sqrt(np.sum(Dx**2, axis=1) + np.sum(Dy**2, axis=1))  
        # return gradients

    def get_gradient_map_sparse(self, xy=None, resolution=200):
        """
        for each pixel, the gradient is calculated by
        Dx(y) = (B(y + (w, 0)) - B(y - (w, 0)))/ 2w 
        Dy(y) = (B(y + (0, h)) - B(y - (0, h)))/ 2h 
        D(y) = squrt(‖Dx(y)‖**2 + ‖Dy(y)‖**2)
        
        where y is a point in the 2D projection space and w and h are a pixel's width and height, respectively.
        B is ppinv.inverse_transform() function.
        """
    
        xy_padding = []
        collection = set()
        for (x, y) in xy:
            collection.add((x, y))
            collection.add((x-1, y))
            collection.add((x+1, y))
            collection.add((x, y-1))
            collection.add((x, y+1))
        
        xy_padding = np.array(list(collection))
        xy_padding_nd = self.get_nd(xy_padding/ resolution)

        dense_map = np.zeros((resolution+2, resolution+2))
        for i, (x, y) in enumerate(xy_padding):
            dense_map[x+1, y+1] = xy_padding_nd[i]

        Dx = (dense_map[2:, 1:-1] - dense_map[:-2, 1:-1]) 
        Dy = (dense_map[1:-1, 2:] - dense_map[1:-1, :-2])

        w = 1/ resolution
        Dx = Dx / (2 * w)
        Dy = Dy / (2 * w)
        # get the gradient norm
        D = np.sqrt(np.sum(Dx**2, axis=1) + np.sum(Dy**2, axis=1))
        ## get only xy in the original xy
        gradients = D[np.array(list(xy))]                               
        return gradients


    def get_gradient_new(self, grid=100):

        x_max, x_min = 1, 0
        y_max, y_min = 1, 0
        pixel_width = (x_max - x_min) / grid
        pixel_height = (y_max - y_min) / grid

        grid_pad = grid + 2 

        xx, yy = np.meshgrid(np.linspace(x_min-pixel_width, x_max+pixel_width, grid_pad), np.linspace(y_min-pixel_height, y_max+pixel_height, grid_pad)) # make it 100*100 to reduce the computation
        xy = np.c_[xx.ravel(), yy.ravel()]
        # get the gradient
        ndgrid_padding = self.get_nd(xy)
        # print(ndgrid_padding.shape)
        ndgrid_padding = ndgrid_padding.reshape(grid_pad, grid_pad, -1)
        ## remove the padding for gradient map. 
        ## This is the inverse porjection for all the pixels. It can be cached for downstream use, such as decision boundary map
        ndgrid = ndgrid_padding[1:-1, 1:-1, :].reshape(grid*grid, -1)

        Dx = ndgrid_padding[2:, 1:-1] - ndgrid_padding[:-2, 1:-1]
        Dy = ndgrid_padding[1:-1, 2:] - ndgrid_padding[1:-1, :-2]

        ### original implementation
        w = 1/ grid
        Dx = Dx / (2 * w)
        Dy = Dy / (2 * w)
        ## just assume the pixel width and height are both 1
        # Dx = Dx / 2
        # Dy = Dy / 2
        # get the gradient norm
        D = np.sqrt(np.sum(Dx**2, axis=2) + np.sum(Dy**2, axis=2))
        return D
    
    def get_label_prob(self, xy):
        nd_data = self.get_nd(xy)
        probs = self.clf.predict_proba(nd_data)
        conf = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)
        return labels, conf
    
    def get_label_roundtrip(self, xy):
        nd_data = self.get_nd(xy)
        if self.P is None:
            nd_data2 = self.ppinv.inverse_transform(self.ppinv.transform(nd_data))
        else:
            nd_data2 = self.ppinv.inverse_transform(self.P.transform(nd_data))
        probs = self.clf.predict_proba(nd_data2)
        conf = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)
        return labels, conf
    
    def get_deepfool(self, xy, labels=None):
        
        assert isinstance(self.clf, nn.Module)
        device = 'cuda' if next(self.clf.parameters()).is_cuda else 'cpu'
        # time0 = time()
        inverted_grid = self.get_nd(xy)
        # time1 = time()
        inverted_grid = T.tensor(inverted_grid).float().to(device)
        # time2 = time()
        
        (
            perturbed,
            orig_class,
            naive_wormhole_data,
        ) = (
            # deepfool_batch(self.clf, inverted_grid, labels)
            # if inverted_grid.is_cuda
            # else 
            deepfool_minibatches(model=self.clf, input_batch=inverted_grid, labels=labels, batch_size=3)
        )
        
        dist_map = (
            T.linalg.norm(inverted_grid - perturbed, dim=-1).detach().cpu().numpy()
        )
        # perturbed = perturbed.cpu().numpy()
        # naive_wormhole_data = naive_wormhole_data.cpu().numpy()
        
        # self.time_acc += time1 - time0
        # self.time_acc2 += time2 - time1
        return dist_map
    
    def get_dist2nearest(self, xy):
        nd_data = self.get_nd(xy)
        dist, _ = self.nnsearcher.kneighbors(nd_data)
        return dist.flatten()
    
    
    def get_map(self, content:str='label', fast_strategy:bool=False, resolution: int=128, interpolation_method: str='linear', initial_resolution:int=32, threshold=0.2) -> tuple:
        # time0 = time()
        # self.time_called = 0
        # self.time_acc = 0
        # self.time_acc2 = 0
        if not fast_strategy:
            print('slow strategy')
            xx, yy = self.make_meshgrid(grid=resolution)
            XY = np.c_[xx.ravel(), yy.ravel()]
            conf = None
            main = None
            sparse = None
            match content:
                case 'label':
                    main, conf = self.get_label_prob(XY)
                case 'label_roundtrip':
                    main, conf = self.get_label_roundtrip(XY)
                case 'gradient':
                    conf =  self.get_gradient_new(grid=resolution)
                case 'dist_map' | 'dist_map_general':
                    conf = self.get_deepfool(XY)
                case 'nearest':
                    self.nnsearcher = NearestNeighbors(n_neighbors=1).fit(self.X)
                    conf = self.get_dist2nearest(XY)
                case _:
                    raise ValueError('content not supported')
            if main is not None:
                main = main.reshape(xx.shape)
                main = np.flip(main.reshape(xx.shape), axis=0)
            if conf is not None:
                conf = conf.reshape(xx.shape)
                conf = np.flip(conf, axis=0)
            
        else:
            print('fast strategy')
            match content:
                case 'label':
                    main, conf, sparse = self.get_fastmap(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content)
                case 'label_roundtrip':
                    main, conf, sparse = self.get_fastmap(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content)
                case 'gradient':
                    main, conf, sparse =  self.get_fastmap_general(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content, threshold=threshold)
                case 'dist_map':
                    main, conf, sparse = self.get_fastmap(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content)
                case 'dist_map_general':
                    main, conf, sparse = self.get_fastmap_general(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content, threshold=threshold)
                case 'gradient_reduced':
                    self.nd_sparse = dict()
                    main, conf, sparse =  self.get_fastmap_general(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content, threshold=threshold)
                case 'nearest':
                    self.nnsearcher = NearestNeighbors(n_neighbors=1).fit(self.X)
                    main, conf, sparse = self.get_fastmap_general(resolution=resolution, computational_budget=None, interpolation_method=interpolation_method, initial_resolution=initial_resolution, content=content, threshold=threshold)
                case _:
                    raise ValueError('content not supported')
            main = np.rot90(main, k=1)
            conf = np.rot90(conf, k=1)
            sparse = np.array(sparse)
        # print('time accumulative:', self.time_acc)
        # print('time accumulative2:', self.time_acc2)
        # print('time called:', self.time_called)
        # print('time total:', time()-time0)
        return (main, conf, sparse)
    

    def get_content_value(self, content:str, space2d):
        conf = None
        match content:
            case 'label':
                main, conf = self.get_label_prob(space2d)
            case 'label_roundtrip':
                main, conf = self.get_label_roundtrip(space2d)
            # case 'gradient':
            #     main, _ = self.get_label_prob(space2d)
            #     conf = self.get_gradient_map_general(space2d)
            case _:
                # print('not label content; return label content instead')
                main, conf = self.get_label_prob(space2d)
        return main, conf
    
    def get_non_label_content(self, content:str, spare_map, resolution:int):
        space2d = spare_map[:, :2].astype(int)
        labels = spare_map[:, 3]
        wh = spare_map[:, 4:6]
        
        scaled_2d = space2d / resolution
        match content:
            case 'gradient':
                value = self.get_gradient_map_general(scaled_2d, resolution=resolution).reshape(-1, 1)
            case 'dist_map':
                value = self.get_deepfool(scaled_2d, labels).reshape(-1, 1)
            case _:
                raise ValueError('content not supported')

        return np.concatenate([space2d, value, labels.reshape(-1, 1), wh], axis=1)


    def get_fastmap(self, resolution: int, computational_budget=None, interpolation_method: str = "linear", initial_resolution: int | None = None, content:str | None =None):
        assert(initial_resolution > 0)
        assert(int(initial_resolution) == initial_resolution)    
        assert(initial_resolution < resolution)
        # ------------------------------------------------------------
        INITIAL_COMPUTATIONAL_BUDGET = computational_budget = resolution * resolution if computational_budget is None else computational_budget

        indexes, sizes, labels, computational_budget, img, confidence_map = self._fill_initial_windows_(
            initial_resolution=initial_resolution, 
            resolution=resolution, 
            computational_budget=computational_budget,
            confidence_interpolation_method=interpolation_method,
            content=content)

        # analyze the initial points and generate the priority queue
        priority_queue = PriorityQueue()
        priority_queue = self._update_priority_queue_(priority_queue, img, indexes, sizes, labels)
        
        # -------------------------------------
        # start the iterative process of filling the image
        # self.console.log(f"Starting the iterative process of refining windows...")
        # self.console.log(f'BINARY SPLIT, interpolation_method: {interpolation_method}')
        # count = 0
        while computational_budget > 0 and not priority_queue.empty():
            # take the highest priority tasks
            items = get_tasks_with_same_priority(priority_queue)

            space2d, indices = [], []
            single_points_space, single_points_indices = [], []
            window_sizes = []

            for (w, h, i, j) in items:
                
                if w == 1 and h == 1:   ## reached the smallest window
                    single_points_space.append(((i+0.0) / resolution, (j+0.0) / resolution))
                    single_points_indices.append((int(i), int(j)))
                    continue
                
                neighbors, sizes = binary_split(i, j, w, h)
                space2d += [(x / resolution, y / resolution) for (x, y) in neighbors]###?????  x and y are switched
                window_sizes += sizes
                indices += neighbors

            space = single_points_space + space2d

            # #####debug
            # space = space2d
            # if space == []:
                # break

            # check if the computational budget is enough and update it
            if computational_budget - len(space) < 0:
                # self.console.warn("Computational budget exceeded, stopping the process")
                break

            # decode the space
            predicted_labels, predicted_confidence = self.get_content_value(content, space)
            
            computational_budget -= len(space)

            single_points_labels = predicted_labels[:len(single_points_space)]
            single_points_confidence = predicted_confidence[:len(single_points_space)]
            predicted_labels = predicted_labels[len(single_points_space):]  ### wocao redefine the predicted_labels
            predicted_confidence = predicted_confidence[len(single_points_space):]
         
            # fill the new image with the new labels and update the priority queue
            for (w, h), (x, y), label, conf in zip(window_sizes, indices, predicted_labels, predicted_confidence):
                # all the pixels in the window are set to the same label
                # starting from the top left corner of the window
                # ending at the bottom right corner of the window
                # the +1 is because the range function is not inclusive
                confidence_map.append((x, y, conf, label, w, h))  ###?????  x and y are switched
                x0, x1, y0, y1 = get_window_borders(x, y, w, h)
                # img[y0:y1 + 1, x0:x1 + 1] = label
                img[x0:x1 + 1, y0:y1 + 1] = label
          
            # fill the new image with the single points  #### Isn't this already done in the previous loop?  
            #### neet to switch x and y
            for i in range(len(single_points_indices)):
                ind_x, ind_y = single_points_indices[i]
                img[ind_x, ind_y] = single_points_labels[i]  ### out of index sometimes
                confidence_map.append((single_points_indices[i][0], single_points_indices[i][1], single_points_confidence[i], single_points_labels[i], 1, 1))

            # update the priority queue
            priority_queue = self._update_priority_queue_(priority_queue, img, indices, window_sizes, predicted_labels)
            # count += 1
            # print(count)
            # if count == 1:
            #     break

        # summary
        # self.console.log(f"Finished decoding the image, initial computational budget: {INITIAL_COMPUTATIONAL_BUDGET} computational budget left: {computational_budget}")
        # self.console.log(f"Items left in the priority queue: {priority_queue.qsize()}")

        ### 
        if 'label' not in content:
            # print(np.array(confidence_map).shape)
            print(f'relace the label with the {content}')
            confidence_map = self.get_non_label_content(content, np.array(confidence_map), resolution)
            # print(confidence_map.shape)

        # generating the confidence image using interpolation based on the confidence map
        img_confidence = self._generate_interpolated_image_(sparse_map=confidence_map,
                                                            resolution=resolution,
                                                            method=interpolation_method).T

        return img, img_confidence, confidence_map


    def _fill_initial_windows_(self, initial_resolution: int, resolution: int, computational_budget: int, confidence_interpolation_method: str = "linear", content=None):
        
        window_size = resolution // initial_resolution
        img = np.zeros((resolution, resolution), dtype=np.int16)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # generate the initial points
        indexes, sizes, border_indexes = generate_windows(window_size, initial_resolution=initial_resolution, resolution=resolution)  ## checked
        # creating an artificial border for the 2D confidence image
        confidence_map = self._generate_confidence_border_(resolution=resolution, border_indexes=border_indexes, content=content) if confidence_interpolation_method != "nearest" else []
        computational_budget -= len(confidence_map) 

        # ------------------------------------------------------------   
        space2d = np.array(indexes) / resolution  
        predicted_labels, predicted_confidence = self.get_content_value(content, space2d)
      
        computational_budget -= len(indexes)

        # fill the initial points in the 2D image
        for (w, h), (x, y), label, conf in zip(sizes, indexes, predicted_labels, predicted_confidence):
            x0, x1, y0, y1 = get_window_borders(x, y, w, h)
            img[x0:x1 + 1, y0:y1 + 1] = label
            confidence_map.append((x, y, conf, label, w, h))
        
        return indexes, sizes, predicted_labels, computational_budget, img, confidence_map

    def _generate_confidence_border_(self, resolution: int, border_indexes, content:str):
        space2d_border = np.array(border_indexes) / resolution

        #####################################3
        labels, confidences_border = self.get_content_value(content, space2d_border)
        confidence_map = [(i, j, conf, label, 0, 0) for (i, j), conf, label in zip(border_indexes, confidences_border, labels)]
        return confidence_map
    
    def get_fastmap_general(self, resolution: int, computational_budget=None, interpolation_method: str = "linear", initial_resolution: int | None = None, content:str | None =None, threshold=0.05):
        assert(initial_resolution > 0)
        assert(int(initial_resolution) == initial_resolution)    
        assert(initial_resolution < resolution)
        # ------------------------------------------------------------
        
        # time0 = time()
        INITIAL_COMPUTATIONAL_BUDGET = computational_budget = resolution * resolution if computational_budget is None else computational_budget

        indexes, sizes, computational_budget, img, confidence_map, diff = self._fill_initial_windows_general_(
            initial_resolution=initial_resolution, 
            resolution=resolution, 
            computational_budget=computational_budget,
            confidence_interpolation_method=interpolation_method,
            content=content,
            )

        threshold_abs = diff * threshold
        print(f'threshold_abs: {threshold_abs}')
        # analyze the initial points and generate the priority queue
        t = np.e**(-1)
        priority_queue = PriorityQueue()
        priority_queue = self._update_priority_queue_general_(priority_queue, img, indexes, sizes, threshold=threshold_abs*t)

        # time1 = time()
        # print(f'initial windows time: {time1 - time0}')
        
        # -------------------------------------
        # start the iterative process of filling the image
        # self.console.log(f"Starting the iterative process of refining windows...")
        # self.console.log(f'BINARY SPLIT, interpolation_method: {interpolation_method}')
        # count = 0
        # self.time_acc = 0
        while computational_budget > 0 and not priority_queue.empty():
            # take the highest priority tasks
            items = get_tasks_with_same_priority(priority_queue)

            space2d, indices = [], []
            single_points_space, single_points_indices = [], []
            window_sizes = []

            for (w, h, i, j) in items:
                
                if w == 1 and h == 1:   ## reached the smallest window
                    single_points_space.append((i / resolution, j / resolution))
                    single_points_indices.append((int(i), int(j)))
                    continue
                
                neighbors, sizes = binary_split(i, j, w, h)
                space2d += [(x / resolution, y / resolution) for (x, y) in neighbors]###?????  x and y are switched
                window_sizes += sizes
                indices += neighbors

            space = single_points_space + space2d

            # #####debug
            # space = space2d
            # if space == []:
                # break

            # check if the computational budget is enough and update it
            if computational_budget - len(space) < 0:
                # self.console.warn("Computational budget exceeded, stopping the process")
                break

            # decode the space
            # time2 = time()
            predicted_values = self.get_content_value_general(content, space, resolution=resolution)
            # time3 = time()
            # time_acc += time3 - time2
            
            computational_budget -= len(space)


            single_points_values = predicted_values[:len(single_points_space)]
            predicted_values = predicted_values[len(single_points_space):]
         
            # fill the new image with the new labels and update the priority queue
            for (w, h), (x, y),  v in zip(window_sizes, indices, predicted_values):
                # all the pixels in the window are set to the same label
                # starting from the top left corner of the window
                # ending at the bottom right corner of the window
                # the +1 is because the range function is not inclusive
                confidence_map.append((x, y, v, w, h))  ###?????  x and y are switched
                x0, x1, y0, y1 = get_window_borders(x, y, w, h)
                # img[y0:y1 + 1, x0:x1 + 1] = label
                img[x0:x1 + 1, y0:y1 + 1] = v
          
            # fill the new image with the single points  #### Isn't this already done in the previous loop?  
            #### neet to switch x and y
            for i in range(len(single_points_values)):
                img[single_points_indices[i]] = single_points_values[i]  ### out of index sometimes
                confidence_map.append((single_points_indices[i][0], single_points_indices[i][1], single_points_values[i], 1, 1))
            
            # update the priority queue
            t = np.e**-(w*initial_resolution/resolution)
            priority_queue = self._update_priority_queue_general_(priority_queue, img, indices, window_sizes, threshold=threshold_abs * t)


        # generating the confidence image using interpolation based on the confidence map
        img_interpolated = self._generate_interpolated_image_(sparse_map=confidence_map,
                                                            resolution=resolution,
                                                            method=interpolation_method).T

 
        return img, img_interpolated, confidence_map
    
    def _fill_initial_windows_general_(self, initial_resolution: int, resolution: int, computational_budget: int, confidence_interpolation_method: str = "linear", content=None):
        
        window_size = resolution // initial_resolution
        img = np.zeros((resolution, resolution), dtype=np.float32)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # generate the initial points
        indexes, sizes, border_indexes = generate_windows(window_size, initial_resolution=initial_resolution, resolution=resolution)  ## checked
        # creating an artificial border for the 2D confidence image
        confidence_map = self._generate_confidence_border_general_(resolution=resolution, border_indexes=border_indexes, content=content) if confidence_interpolation_method != "nearest" else []
        computational_budget -= len(confidence_map) 

        # ------------------------------------------------------------   
        space2d = np.array(indexes) / resolution  
        predicted_values = self.get_content_value_general(content, space2d, resolution=resolution)

        diff = predicted_values.max() - predicted_values.min()
      
        computational_budget -= len(indexes)

        # fill the initial points in the 2D image
        for (w, h), (x, y), v in zip(sizes, indexes, predicted_values):
            x0, x1, y0, y1 = get_window_borders(x, y, w, h)
            img[x0:x1 + 1, y0:y1 + 1] = v
            confidence_map.append((x, y, v, w, h))
        
        return indexes, sizes, computational_budget, img, confidence_map, diff

    def _generate_confidence_border_general_(self, resolution: int, border_indexes, content:str):
        space2d_border = np.array(border_indexes) / resolution

        #####################################3
        values = self.get_content_value_general(content, space2d_border, resolution)
        confidence_map = [(i, j, v, 0, 0) for (i, j), v, in zip(border_indexes, values)]
        return confidence_map
    
    def get_content_value_general(self, content:str, space2d, resolution:int):
        conf = None
        match content:
            case 'label':
                values, _ = self.get_label_prob(space2d)
            case 'label_roundtrip':
                values, _ = self.get_label_roundtrip(space2d)
            case 'gradient':
                # main, _ = self.get_label_prob(space2d)
                values = self.get_gradient_map_general(space2d, resolution=resolution)
            case 'gradient_reduced':
                values = self.get_gradient_map_reduced(space2d, resolution=resolution)
            case 'dist_map' | 'dist_map_general':
                values = self.get_deepfool(space2d)
            case 'nearest':
                values = self.get_dist2nearest(space2d)
            case _:
                raise ValueError('content not supported')
            # case _:
                # print('not label content; return label content instead')
                # main, conf = self.get_label_prob(space2d)
        return values
    
    def _update_priority_queue_general_(self, priority_queue, img, indexes, sizes, threshold):
        for (w, h), (x, y) in zip(sizes, indexes):
            priority = get_pixel_priority_general(img, x, y, w, h, threshold)  ## updated
            if priority != -1:
                priority_queue.put((priority, (w, h, x, y)))    ## updated
        return priority_queue
    
    def _update_priority_queue_(self, priority_queue, img, indexes, sizes, labels):
        for (w, h), (x, y), label in zip(sizes, indexes, labels):
            priority = get_pixel_priority(img, x, y, w, h, label)  ## updated
            if priority != -1:
                priority_queue.put((priority, (w, h, x, y)))    ## updated
        return priority_queue
    
    def _generate_interpolated_image_(self, sparse_map, resolution:int, method:str='linear'):
        """
        A private method that uses interpolation to generate the values for the 2D space image
           The sparse map is a list of tuples (x, y, data)
           The sparse map represents a structured but non uniform grid of data values
           Therefore usual rectangular interpolation methods are not suitable
           For the interpolation we use the scipy.interpolate.griddata function with the linear method
        Args:
            sparse_map (list): a list of tuples (x, y, data) where x and y are the coordinates of the pixel and data is the data value
            resolution (int): the resolution of the image we want to generate (the image will be a square image)
            method (str, optional): The method to be used for the interpolation. Defaults to 'linear'. Available methods are: 'nearest', 'linear', 'cubic'

        Returns:
            np.array: an array of shape (resolution, resolution) containing the data values for the 2D space image
        """
        X, Y, Z = [], [], []
        for item in sparse_map:
            X.append(item[0])
            Y.append(item[1])
            Z.append(item[2])
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        xi = np.linspace(0, resolution-1, resolution)
        yi = np.linspace(0, resolution-1, resolution)
        return interpolate.griddata((X, Y), Z, (xi[None, :], yi[:, None]), method=method)


    def plot_gradient_map(self, ax=None, cmap=None, grid=256, fast=False, initial_resolution=32, interpolation_method='linear', threshold=0.125, reduced=False, plot_mean=True):
        """Plot probability map for the classifier
        """
        if ax is None:
            ax = plt.gca()
        if reduced:
            _, D, sparse = self.get_map(content='gradient_reduced', fast_strategy=fast, resolution=grid, initial_resolution=initial_resolution, interpolation_method=interpolation_method, threshold=threshold) 
        else:
            _, D, sparse = self.get_map(content='gradient', fast_strategy=fast, resolution=grid, initial_resolution=initial_resolution, interpolation_method=interpolation_method, threshold=threshold) 
            
        if cmap is None:
            CMAP = cm.get_cmap('magma')
        else:
            CMAP = colormaps[cmap]

        ax.imshow(D, cmap=CMAP, extent=[0, 1, 0, 1])

        ## plot mean of D as text on the plot
        if plot_mean:
            ax.text(0.1, 0.9, f'{np.mean(D):.4f}', fontsize=12, color='w', ha='center', va='center')  

        ax.set_xticks([])
        ax.set_yticks([])
        # aspect square
        # ax.set_aspect('equal')
        return ax, sparse

    
    def plot_boundary(self, ax=None, fast=False, grid=256):
        """Plot probability map for the classifier
        """
        labels, _, sparse = self.get_map(content='label', fast_strategy=fast, resolution=grid)
        labels = np.flip(labels, axis=0)
        if ax is None:
            ax = plt.gca()
        xx, yy = self.make_meshgrid(grid=grid)
        # labels = labels.reshape(xx.shape)
        ax.contour(xx, yy, labels.reshape(xx.shape), levels=(np.arange(self.clf.classes_.max() + 2) - 0.5),
                    linewidths=1, colors="k", antialiased=True)
        return ax, sparse
    
    def plot_dist_map(self, ax=None, cmap='viridis', grid=256, fast=False, initial_resolution=32, content='dist_map_general', threshold=0.15, plot_boundary=False, plot_mean=False):
        """Plot probability map for the classifier
        """
        if ax is None:
            ax = plt.gca()

        _, D, sparse = self.get_map(content=content, fast_strategy=fast, resolution=grid, initial_resolution=initial_resolution, threshold=threshold,)
        CMAP = colormaps[cmap]
        ax.imshow(D, cmap=cmap, extent=[0, 1, 0, 1])  

        if plot_mean:
            ax.text(0.1, 0.9, f'{np.mean(D):.4f}', fontsize=12, color='w', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        # aspect square
        # ax.set_aspect('equal')
        if plot_boundary:
            ax = self.plot_boundary(ax=ax, fast=fast, grid=grid)
        
        return ax, sparse
    
    # def plot_round_label_map(self, ax=None, cmap='tab10', grid=200, fast=False, initial_resolution=32):
    #     labels, alpha, sparse = self.get_map(content='label_roundtrip', fast_strategy=fast, resolution=grid, initial_resolution=initial_resolution, )
    #     if ax is None:
    #         ax = plt.gca()
    #     labels_normlized = labels/self.clf.classes_.max() if self.clf.classes_.max() > 9 else labels/9
    #     CMAP = colormaps[cmap]
    #     map = CMAP(labels_normlized)
    #     ax.imshow(map, interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    #     # if proba:
    #     #     ax.set_facecolor('black')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     # set lim
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     return ax, sparse

    def plot_decision_map(self, ax=None, cmap='tab10', epsilo=0.9, proba=True, fast=False, grid=200, initial_resolution=32, interpolation_method='linear', content='label'):
        """Plot probability map for the classifier
        """
        labels, alpha, sparse = self.get_map(content=content, fast_strategy=fast, resolution=grid, initial_resolution=initial_resolution, interpolation_method=interpolation_method)
        if ax is None:
            ax = plt.gca()

        labels_normlized = labels/self.clf.classes_.max() if self.clf.classes_.max() > 9 else labels/9
        CMAP = colormaps[cmap]
        map = CMAP(labels_normlized)
        if proba:
            alpha = alpha.reshape(-1, 1)
            alpha = minmax_scale(alpha, feature_range=((1/self.clf.classes_.max()), epsilo))
            alpha = alpha.reshape(grid, grid)
            map[:, :, 3] = alpha   # plus a float to control the transparency
        map[:, :, 3] *= epsilo  # plus a float to control the transparency
      
        ax.imshow(map, interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
        # if proba:
        #     ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
        # set lim
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax, sparse

    def plot_training_data(self, ax=None, cmap='tab10', size=10, **kwargs):
        """Plot probability map for the classifier
        """
        if ax is None:
            ax = plt.gca()
        X_2d = self.X2d_scaled
        if self.y_label is not None:
            CMAP = colormaps[cmap]
            lable_norm = self.y_label/self.clf.classes_.max() if self.clf.classes_.max() > 9 else self.y_label/9
            colors = CMAP(lable_norm)
            for i in set(self.y_label):
                ax.scatter(X_2d[self.y_label==i, 0], X_2d[self.y_label==i, 1], marker='.', s=size, edgecolors=None, c=colors[self.y_label==i], label=i, **kwargs)
            ax.legend()
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], marker='.', s=size, edgecolors=None, c='w', **kwargs)
        return ax
    
    def plot_map(self, ax=None, content='label', fast=False, grid=128, B=32):
        match content:
            case 'label':
                ax, sparse = self.plot_decision_map(ax=ax, fast=fast, grid=grid, initial_resolution=B, content='label')
            case 'gradient':
                ax, sparse = self.plot_gradient_map(ax=ax, fast=fast, grid=grid, initial_resolution=B)
            case 'boundary':
                ax, sparse = self.plot_boundary(ax=ax, fast=fast, grid=grid)
            case 'label_roundtrip':
                ax, sparse = self.plot_decision_map(ax=ax, fast=fast, grid=grid, content='label_roundtrip', proba=False, initial_resolution=B)
            case 'dist_map_general':
                ax, sparse = self.plot_dist_map(ax=ax, fast=fast, grid=grid, content='dist_map_general')
            case 'nearest':
                ax, sparse = self.plot_dist_map(ax=ax, fast=fast, grid=grid, content='nearest')
            case _:
                raise ValueError('content should be one of the following: label, gradient, boundary, label_roundtrip, dist_map_general')

        return ax, sparse
    
    
######################################################
    # def inversed_feature(self, scalernd=None):
    #     xx, yy = self.xx, self.yy
    #     scaled_2d = self.scaler2d.inverse_transform(np.c_[xx.ravel(), yy.ravel()]).astype('float32')
    #     inversed = self.projecter.inverse_transform(scaled_2d).astype('float32')
    #     # labels = self.clf.predict(inversed)
    #     if scalernd:
    #         inversed = scalernd.inverse_transform(inversed).astype('float32')
    #     # labels_projecter = self.projecter.predict2d(scaled_2d)

    #     # num_features = inversed.shape[1]
    #     res = inversed.reshape(xx.shape[0], xx.shape[1], -1)
    #     # labels = labels.reshape(xx.shape[0], xx.shape[1])
    #     # labels_projecter = labels_projecter.reshape(xx.shape[0], xx.shape[1])
    #     return res #, labels #labels_projecter

    # def plot_inversed_feature(self, ind, feature_names, ax=None, scalernd=None, countour=True, **params): ##
    #     if not ax:
    #         ax = plt.gca()
    #     fig = ax.get_figure()

    #     res = self.inversed_feature(scalernd=scalernd)
    #     labels = self.map_res[1].reshape(self.xx.shape[0], self.xx.shape[1])

    #     xx, yy = self.xx, self.yy
        
    #     feature_plot = 10 ** res[:, :, ind] -1
    #     # feature_plot = np.flipud(feature_plot)
    #     if countour:
    #         temmap = ax.contourf(xx, yy, feature_plot, cmap='bwr', norm=colors.LogNorm(vmin=feature_plot.min(), vmax=feature_plot.max()),  **params) #
            
    #     else:
    #         temmap = ax.imshow(np.flip(feature_plot, 0), cmap='bwr', norm=colors.LogNorm(vmin=feature_plot.min(), vmax=feature_plot.max()), extent=[xx.min(), xx.max(), yy.min(), yy.max()])
    #         # ax.invert_yaxis()
    #     cnt = ax.contour(xx, yy, labels, **params, colors='k')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title('{}'.format(feature_names[ind]))
    #     # invert y axis
    #     # color bar
    #     cbar = fig.colorbar(temmap, ax=ax)
    #     return ax
#########################################################

    def transform(self, X, normed=True):
        X2d = self.ppinv.transform(X)
        X2d = self.scaler2d.transform(X2d)
        return X2d