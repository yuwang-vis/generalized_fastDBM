"""
code from the paper: 
Blumberg, D., Wang, Y., Telea, A., Keim, D. A., & Dennig, F. L. (2024). Inverting Multidimensional Scaling Projections Using Data Point Multilateration. EuroVis Workshop on Visual Analytics (EuroVA).
"""

import numpy as np
from scipy.linalg import null_space
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances

# import os  
# os.chdir(os.path.dirname(__file__))


def euclidean_mds(data):
    mds = MDS()
    transformed = mds.fit_transform(data)
    stress = mds.stress_
    return transformed, stress

#NN calculation
def get_nearest_neighbor_hd(point_hd, data, reduced_data):
    point_hd = point_hd.reshape(1, -1) ###
    dist = np.sum((data - point_hd)**2, axis=1)
    index_nn = np.argmin(dist)
    return data[index_nn], reduced_data[index_nn]

def get_nearest_neighbor_ld(point_ld, data, reduced_data):
    dist = np.sum((reduced_data - point_ld)**2, axis=1)
    index_nn = np.argmin(dist)
    return data[index_nn], reduced_data[index_nn]

#point selection strategies for multilateration
def get_furthest_points(data, reduced_data, p):
    '''
    function to get the n+1 furthest points and their distances (in reduced space) 
    to the point p 
    
    '''
    n = data.shape[1] #dimensionality in high dimensional space
    d = euclidean_distances(reduced_data, p) #distances to p
    combine = np.hstack((data,d)) 
    sorted_combine = combine[np.argsort(combine[:,-1])] #sort according to distances
    #return n+1 furthest points
    furthest_d = sorted_combine[-(n+1):,-1].reshape(-1,1) 
    furthest_P = sorted_combine[-(n+1):,:-1] 
    return furthest_P, furthest_d

def get_nearest_points(data, reduced_data, p):
    '''
    function to get the n+1 nearest points and their distances (in reduced space) 
    to the point p 
    
    '''
    n = data.shape[1] #dimensionality in high dimensional space
    d = euclidean_distances(reduced_data, p) #distances to p
    combine = np.hstack((data,d)) 
    sorted_combine = combine[np.argsort(combine[:,-1])] #sort according to distances
    #return n+1 nearest points (exluding p itself if p is existing point)
    if np.any([np.array_equal(p, point) for point in reduced_data]):
        nearest_d = sorted_combine[1:n+2, -1].reshape(-1, 1) # exclude p
        nearest_P = sorted_combine[1:n+2, :-1]
    else:
        nearest_d = sorted_combine[:n+1,-1].reshape(-1,1) 
        nearest_P = sorted_combine[:n+1,:-1] 
    return nearest_P, nearest_d

def get_random_points(data, reduced_data, p):
    '''
    function to get random n+1 points and their distances (in reduced space) 
    to the point p 
    
    '''
    # print(data.shape)
    n = data.shape[1] #dimensionality in high dimensional space
    #return n+1 random points
    random_indices = np.random.choice(data.shape[0], n+1, replace=False) # randomly select indices
    while np.any([np.array_equal(p, point) for point in reduced_data[random_indices]]): # ensure p is not a selected point 
        random_indices = np.random.choice(data.shape[0], n+1, replace=False)
    random_P = data[random_indices] 
    random_d = euclidean_distances(reduced_data[random_indices], p).reshape(-1, 1) # distances to p
    return random_P, random_d
    
 
#multilateration    
def multilateration(P,d):
    '''
    function to get the position of a point by using the distances d of the
    points P to this point

    '''
    A = (P[1:,:] - P[0,:]) * (-2)
    e = d**2 - np.sum(P**2, axis = 1).reshape(-1,1)
    B = e[1:] - e[0]
    if (np.linalg.det(A) != 0): #full rank -> only 1 exact solution
        p_particular = np.linalg.solve(A, B).T
        null_space_A = np.nan
    else: #solution with min l2 norm
        p_particular = np.linalg.lstsq(A.T, B, rcond=None)[0] #B.T
        null_space_A = null_space(A.T)
        #all solutions: p_particular + c * null_space
    return p_particular, null_space_A

#inverse projection of existing data point and any point
def get_high_dimensional_position(data, reduced_data, i, point_selection='random', trials = 20):
    '''
    function to estimate the position of a point in n-dimensional space
    using multilateration

    Parameters
    ----------
    data : ndarray
        data in high-dimensional space.
    reduced_data : ndarray
        data in low-dimensional space.
    i : int
        index of point in dataset.
    point_selection: str
        point selection strategy for multilateration. Default is random.
    trials: int
        number of trials for the random point selection strategy. Default is 20.

    Returns
    -------
    p : np.array - (n x 1) vector
        real position of point in high-dimensional space.
    position : np.array - (n x 1) vector
        estimated position of point in high-dimensional space.

    '''
    p = data[i,:] #point of interest in high dimensional space
    p_reduced = reduced_data[i,:].reshape(1,-1) #point of interest in low dimensional space
    #get n+1 points according to point selection strategy
    if point_selection == 'furthest':
        P, d = get_furthest_points(data, reduced_data, p_reduced)
        position, _ = multilateration(P, d)
    elif point_selection == 'nearest':
        P, d = get_nearest_points(data, reduced_data, p_reduced)
        position, _ = multilateration(P, d)
    else: #random
        positions = []
        for i in range(trials):
            P, d = get_random_points(data, reduced_data, p_reduced)
            random_position, _ = multilateration(P, d)
            positions.append(random_position.flatten())
        positions = np.array(positions)
        #position = np.mean(positions, axis=0) # get centroid of randomly generated positions
        position = np.median(positions, axis=0) # get median of randomly generated positions
    return p, position


def get_any_high_dimensional_position(data, reduced_data, point, point_selection='random', trials = 20):
    '''
    function to estimate the position of a point in n-dimensional space
    using multilateration

    Parameters
    ----------
    data : ndarray
        data in high-dimensional space.
    reduced_data : ndarray
        data in low-dimensional space.
    point : arry
        arbitrary point in low-dimensional space.
    point_selection: str
        point selection strategy for multilateration. Default is random.
    trials: int
        number of trials for the random point selection strategy. Default is 20.

    Returns
    -------
    position : np.array - (n x 1) vector
        estimated position of point in high-dimensional space.

    '''
    # print(data.shape, reduced_data.shape, point.shape)
    point = point.reshape(1,-1)
    #get n+1 points according to point selection strategy
    if point_selection == 'furthest':
        P, d = get_furthest_points(data, reduced_data, point)
        position, _ = multilateration(P, d)
    elif point_selection == 'nearest':
        P, d = get_nearest_points(data, reduced_data, point)
        position, _ = multilateration(P, d)
    else: #random
        positions = []
        for i in range(trials):
            P, d = get_random_points(data, reduced_data, point)
            random_position, _ = multilateration(P, d)
            positions.append(random_position.flatten())
        positions = np.array(positions)
        #position = np.mean(positions, axis=0) # get centroid of randomly generated positions
        position = np.median(positions, axis=0) # get median of randomly generated positions
    return position



class MDSinv:
    def __init__(self,  point_selection='random', trials = 20) -> None:
        self.point_selection = point_selection
        self.trials = trials

    def fit(self, X2d, X, **kwarg):
        self.X = X
        self.X2d = X2d

    def transform(self, p_list, **kwarg):
        # p_list = np.array(p_list)
        # v_func = np.vectorize(get_any_high_dimensional_position)
        # return v_func(self.X, self.X2d, p_list, self.point_selection, self.trials)
        Xnd_recons = []
        for p in p_list:
            pnd = get_any_high_dimensional_position(self.X, self.X2d, p, self.point_selection, self.trials)
            Xnd_recons.append(pnd.reshape(-1))
        return np.array(Xnd_recons).reshape(-1, self.X.shape[1])
    
    def inverse_transform(self, p, **kwarg):
        return self.transform(p, **kwarg)
    
    def reset(self):
        self.X = None
        self.X2d = None
        
    def find_z(self, dummy):
        return dummy


#test
'''
df = pd.read_csv('iris.data')
data = df.iloc[:,:-1].to_numpy()
reduced_data, stress = euclidean_mds(data)
x, position, null_space_A = get_high_dimensional_position(data, reduced_data, 0)
print(f'real position: {x}')
print(f'estimated position: {position}')
print(f'null space: {null_space_A}')

#%%test of multilateration function
def get_points(n):
    array = np.random.randint(10, size=(n+2, n))
    P = array[:-1,:]
    x = array[-1,:].reshape(1, -1)
    d = euclidean_distances(P, x)
    return P, d, x

P,d,x = get_points(4)
        
estimated_x, null_space_A = multilateration(P,d)

print(f'real position: {x}')
print(f'estimated position: {estimated_x}')
print(f'null space: {null_space_A}')

#%% test of null space
A = np.array([[1,1],[0,0]])
B = np.array([5,0])
x = np.linalg.lstsq(A,B,rcond=None)[0] #singular matrix error if np.linalg.solve
print(x) 
null_space_A = null_space(A)
print(null_space_A)
#all solutions: x + c * null_space
'''
