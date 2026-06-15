__authors__ = [str(1745518), str(1748251), str(1746018)]
__group__ = 'Team_64'

import numpy as np
from scipy.spatial.distance import cdist


class KNN:

    def __init__(self, train_data, labels, options=None):
        self._init_options(options)
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_options(self, options=None):
        if options is None:
            options = {}

        if 'feature_type' not in options:
            options['feature_type'] = 'pixels'

        if 'distance' not in options:
            options['distance'] = 'euclidean'

        if 'voting' not in options:
            options['voting'] = 'normal'

        self.options = options

    def extract_features(self, data):
        data = np.array(data, dtype=float)

        if self.options['feature_type'] == 'pixels':
            return data.reshape(data.shape[0], -1)

        elif self.options['feature_type'] == 'simple':
            features = []

            for img in data:
                if img.ndim == 3:
                    gray = np.mean(img, axis=2)
                else:
                    gray = img

                if gray.ndim == 1:
                    gray = gray.reshape(1, -1)

                h, w = gray.shape

                mitjana = np.mean(gray)
                desviacio = np.std(gray)

                superior = np.mean(gray[:h // 2, :]) if h > 1 else mitjana
                inferior = np.mean(gray[h // 2:, :]) if h > 1 else mitjana
                esquerra = np.mean(gray[:, :w // 2]) if w > 1 else mitjana
                dreta = np.mean(gray[:, w // 2:]) if w > 1 else mitjana

                features.append([
                    mitjana,
                    desviacio,
                    superior,
                    inferior,
                    esquerra,
                    dreta
                ])

            return np.array(features)

        elif self.options['feature_type'] == 'grid':
            features = []

            for img in data:
                if img.ndim == 3:
                    gray = np.mean(img, axis=2)
                else:
                    gray = img

                h, w = gray.shape
                img_features = []

                for i in range(3):
                    for j in range(3):
                        y1 = i * h // 3
                        y2 = (i + 1) * h // 3
                        x1 = j * w // 3
                        x2 = (j + 1) * w // 3

                        zone = gray[y1:y2, x1:x2]
                        img_features.append(np.mean(zone))

                img_features.append(np.mean(gray))
                img_features.append(np.std(gray))

                features.append(img_features)

            return np.array(features)

        else:
            return data.reshape(data.shape[0], -1)

    def _init_train(self, train_data):
        self.train_data = self.extract_features(train_data)

    def get_k_neighbours(self, test_data, k):
        test_data = self.extract_features(test_data)

        metric = self.options['distance']

        if metric == 'manhattan':
            distances = cdist(test_data, self.train_data, metric='cityblock')

        elif metric == 'cosine':
            distances = cdist(test_data, self.train_data, metric='cosine')

        else:
            distances = cdist(test_data, self.train_data, metric='euclidean')

        nearest = np.argsort(distances, axis=1)[:, :k]

        self.neighbors = self.labels[nearest]
        self.neighbor_distances = np.take_along_axis(distances, nearest, axis=1)

    def get_class(self):
        classes = []

        for row in self.neighbors:
            values, first, counts = np.unique(row, return_index=True, return_counts=True)

            max_count = np.max(counts)
            tied = np.where(counts == max_count)[0]

            winner = values[tied[np.argmin(first[tied])]]
            classes.append(winner)

        return np.array(classes)

    def get_class_with_confidence(self):
        """
        Retorna:
            classes: predicció final
            confidences: confiança entre 0 i 1

        Si voting = normal:
            confiança = vots de la classe guanyadora / k

        Si voting = weighted:
            confiança = pes de la classe guanyadora / pes total
        """
        classes = []
        confidences = []

        for i in range(len(self.neighbors)):

            if self.options['voting'] == 'weighted':
                votes = {}

                for j in range(len(self.neighbors[i])):
                    label = self.neighbors[i][j]
                    dist = self.neighbor_distances[i][j]

                    weight = 1 / (dist + 1e-6)

                    if label not in votes:
                        votes[label] = 0

                    votes[label] += weight

                winner = max(votes, key=votes.get)
                total_weight = sum(votes.values())
                confidence = votes[winner] / total_weight

            else:
                row = self.neighbors[i]
                values, first, counts = np.unique(row, return_index=True, return_counts=True)

                max_count = np.max(counts)
                tied = np.where(counts == max_count)[0]

                winner = values[tied[np.argmin(first[tied])]]
                confidence = max_count / len(row)

            classes.append(winner)
            confidences.append(confidence)

        return np.array(classes), np.array(confidences)

    def predict(self, test_data, k):
        self.get_k_neighbours(test_data, k)

        if self.options['voting'] == 'weighted':
            classes, confidences = self.get_class_with_confidence()
            return classes

        return self.get_class()

    def predict_with_confidence(self, test_data, k):
        self.get_k_neighbours(test_data, k)
        return self.get_class_with_confidence()