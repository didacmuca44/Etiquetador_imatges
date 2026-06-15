__authors__ = [str(1745518), str(1748251), str(1746018)]
__group__ = 'Team_64'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            self.X = X.reshape(-1, 1)
        elif X.ndim == 2:
            self.X = X
        else:
            self.X = X.reshape(-1, X.shape[-1])

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=float)
        mode = self.options['km_init'].lower()
    
        if mode == 'first':
            unique_values, first_indices = np.unique(self.X, axis=0, return_index=True)
            first_indices = np.sort(first_indices)
            self.centroids = self.X[first_indices[:self.K]].astype(float)
    
        elif mode == 'random':
            random_indices = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[random_indices].astype(float)
    
        else:  # custom
            mins = np.min(self.X, axis=0)
            maxs = np.max(self.X, axis=0)
    
            if self.K == 1:
                self.centroids = ((mins + maxs) / 2).reshape(1, -1)
            else:
                t = np.linspace(0, 1, self.K).reshape(-1, 1)
                self.centroids = mins + t * (maxs - mins)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids.copy()

        for i in range(self.K):
            points = self.X[self.labels == i]
        
            if len(points) > 0:
                self.centroids[i] = np.mean(points, axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.all(np.abs(self.centroids - self.old_centroids) <= self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        self.num_iter = 0
    
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
    
            if self.converges():
                break
    
        self.get_labels()

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        assigned_centroids = self.centroids[self.labels]
        diff = self.X - assigned_centroids
        self.WCD = np.sum(diff ** 2) / self.X.shape[0]
        return self.WCD

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        threshold = 20
        previous_wcd = None
        best_k = max_K
    
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            current_wcd = self.withinClassDistance()
    
            if previous_wcd is not None:
                dec = 100 * current_wcd / previous_wcd
    
                if 100 - dec < threshold:
                    best_k = k - 1
                    break
    
            previous_wcd = current_wcd
    
        self.K = best_k


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    X = np.array(X, dtype=float)
    C = np.array(C, dtype=float)

    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2))
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    centroids = np.array(centroids, dtype=float)
    probs = utils.get_color_prob(centroids)
    indices = np.argmax(probs, axis=1)
    return list(utils.colors[indices])
