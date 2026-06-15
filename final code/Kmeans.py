__authors__ = [str(1745518), str(1748251), str(1746018)]
__group__ = 'Team_64'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of clusters
                 options (dict): dictionary with options
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        """
        Initialization of all pixels.
        Converts X into a PxD matrix.
        """
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            self.X = X.reshape(-1, 1)

        elif X.ndim == 2:
            self.X = X

        else:
            self.X = X.reshape(-1, X.shape[-1])

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined.
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
            options['fitting'] = 'WCD'

        if 'threshold' not in options:
            options['threshold'] = 20

        self.options = options

    def _init_centroids_kmeanspp(self):
        """
        Inicialització KMeans++.
        Primer centroide aleatori i els següents allunyats dels anteriors.
        """
        centroids = []

        first = np.random.randint(self.X.shape[0])
        centroids.append(self.X[first])

        while len(centroids) < self.K:
            C = np.array(centroids)
            dist = distance(self.X, C)

            min_dist = np.min(dist, axis=1)
            prob = min_dist ** 2

            if np.sum(prob) == 0:
                idx = np.random.randint(self.X.shape[0])
            else:
                prob = prob / np.sum(prob)
                idx = np.random.choice(self.X.shape[0], p=prob)

            centroids.append(self.X[idx])

        self.centroids = np.array(centroids, dtype=float)

    def _init_centroids(self):
        """
        Initialization of centroids.
        Options:
            first
            random
            kmeans++
            custom
        """
        self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=float)
        mode = self.options['km_init'].lower()

        if mode == 'first':
            unique_values, first_indices = np.unique(self.X, axis=0, return_index=True)
            first_indices = np.sort(first_indices)

            if len(first_indices) >= self.K:
                self.centroids = self.X[first_indices[:self.K]].astype(float)
            else:
                random_indices = np.random.choice(self.X.shape[0], self.K, replace=True)
                self.centroids = self.X[random_indices].astype(float)

        elif mode == 'random':
            random_indices = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[random_indices].astype(float)

        elif mode == 'kmeans++':
            self._init_centroids_kmeanspp()

        else:
            mins = np.min(self.X, axis=0)
            maxs = np.max(self.X, axis=0)

            if self.K == 1:
                self.centroids = ((mins + maxs) / 2).reshape(1, -1)
            else:
                t = np.linspace(0, 1, self.K).reshape(-1, 1)
                self.centroids = mins + t * (maxs - mins)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X.
        """
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)

    def get_centroids(self):
        """
        Calculates new centroids based on the assigned points.
        """
        self.old_centroids = self.centroids.copy()

        for i in range(self.K):
            points = self.X[self.labels == i]

            if len(points) > 0:
                self.centroids[i] = np.mean(points, axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids.
        """
        return np.all(np.abs(self.centroids - self.old_centroids) <= self.options['tolerance'])

    def fit(self):
        """
        Runs KMeans until convergence or max_iter.
        """
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
        Returns the within class distance of the current clustering.
        """
        assigned_centroids = self.centroids[self.labels]
        diff = self.X - assigned_centroids
        self.WCD = np.sum(diff ** 2) / self.X.shape[0]
        return self.WCD

    def interClassDistance(self):
        """
        Distància mitjana entre centroides.
        Com més gran, més separats estan els clusters.
        """
        if self.K < 2:
            return 0

        total = 0
        count = 0

        for i in range(self.K):
            for j in range(i + 1, self.K):
                total += np.linalg.norm(self.centroids[i] - self.centroids[j])
                count += 1

        if count == 0:
            return 0

        return total / count

    def fisherCoefficient(self):
        """
        Coeficient de Fisher simple:
            interClassDistance / withinClassDistance
        Com més gran, millor.
        """
        wcd = self.withinClassDistance()
        icd = self.interClassDistance()

        if wcd == 0:
            return 0

        return icd / wcd

    def find_bestK(self, max_K):
        """
        Sets the best K analysing the results up to max_K.
        Options:
            fitting = 'WCD'
            fitting = 'ICD'
            fitting = 'Fisher'
        """
        method = self.options['fitting']
        threshold = self.options['threshold']

        best_k = 2
        best_score = None
        previous_wcd = None

        for k in range(2, max_K + 1):
            self.K = k
            self.fit()

            wcd = self.withinClassDistance()

            if method == 'WCD':
                if previous_wcd is not None:
                    dec = 100 * wcd / previous_wcd

                    if 100 - dec < threshold:
                        best_k = k - 1
                        break

                previous_wcd = wcd
                best_k = k

            elif method == 'ICD':
                score = self.interClassDistance()

                if best_score is None or score > best_score:
                    best_score = score
                    best_k = k

            elif method == 'Fisher':
                score = self.fisherCoefficient()

                if best_score is None or score > best_score:
                    best_score = score
                    best_k = k

            else:
                best_k = k

        self.K = best_k
        self.fit()


def distance(X, C):
    """
    Calculates the distance between each point in X and each centroid in C.

    Args:
        X: PxD array
        C: KxD array

    Returns:
        dist: PxK array
    """
    X = np.array(X, dtype=float)
    C = np.array(C, dtype=float)

    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2))

    return dist


def get_colors(centroids):
    """
    For each centroid returns one of the 11 basic colors.
    """
    centroids = np.array(centroids, dtype=float)

    probs = utils.get_color_prob(centroids)
    indices = np.argmax(probs, axis=1)

    return list(utils.colors[indices])