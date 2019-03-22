import numpy as np


class KNN_classifier:

    def __init__(self, color_features, gradient_features, labels, k, distance_measure="l1", feature="combined"):
        self.color_features = np.array(color_features)
        self.gradient_features = np.array(gradient_features)
        self.labels = np.array(labels)
        self.k = k
        self.distance_measure = distance_measure
        self.feature = feature

    def predict(self, color_feature, gradient_feature):
        if self.distance_measure == "l1":
            if self.feature == "combined":
                color_distances = np.sum(np.abs(self.color_features - color_feature), axis=-1)
                gradient_distances = np.sum(np.abs(self.gradient_features - gradient_feature), axis=-1)
                distances = (color_distances + gradient_distances) / 2
            elif self.feature == "color":
                distances = np.sum(np.abs(self.color_features - color_feature), axis=-1)
            elif self.feature == "gradient":
                distances = np.sum(np.abs(self.gradient_features - gradient_feature), axis=-1)

        elif self.distance_measure == "l2":
            if self.feature == "combined":
                color_distances = np.sqrt(np.sum(np.square(self.color_features - color_feature), axis=-1))
                gradient_distances = np.sqrt(np.sum(np.square(self.gradient_features - gradient_feature), axis=-1))
                distances = (color_distances + gradient_distances) / 2
            elif self.feature == "color":
                distances = np.sqrt(np.sum(np.square(self.color_features - color_feature), axis=-1))
            elif self.feature == "gradient":
                distances = np.sqrt(np.sum(np.square(self.gradient_features - gradient_feature), axis=-1))

        closest_labels = self.labels[np.argsort(distances)[:self.k]]
        labels, counts = np.unique(closest_labels, return_counts=True)
        majority_vote = labels[np.argmax(counts)]
        return majority_vote

    def score(self, color_test, gradient_test, test_classes):
        predictions = []
        for color_feature, gradient_feature in zip(color_test, gradient_test):
            prediction = self.predict(color_feature, gradient_feature)
            predictions.append(prediction)
        return np.sum(np.array(predictions) == np.array(test_classes)) / len(predictions), predictions
