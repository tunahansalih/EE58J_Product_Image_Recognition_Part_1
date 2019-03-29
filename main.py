import json
import os
import pickle
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from image_description import grid_images, get_gradient_orientation_histogram_vector, get_color_histogram_vector
from classifier import KNN_classifier

from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

data_path = "/home/tunahansalih/PycharmProjects/EE58J_Assignment_1/data/SKU_Recognition_Dataset"
pickle_path = "/home/tunahansalih/PycharmProjects/EE58J_Assignment_1/pickles"

images = []
coarse_labels = []
fine_labels = []
for coarse_label in os.listdir(data_path):
    current_coarse_dir = os.path.join(data_path, coarse_label)
    for fine_label in os.listdir(current_coarse_dir):
        current_fine_dir = os.path.join(current_coarse_dir, fine_label)
        for image_name in os.listdir(current_fine_dir):
            if os.path.splitext(image_name)[1] == ".jpg":
                image_path = os.path.join(current_fine_dir, image_name)
                print(image_path)
                image = cv2.imread(image_path)
                images.append(cv2.resize(image, (128, 128)))
                coarse_labels.append(coarse_label)
                fine_labels.append(fine_label)
            else:
                print("Not a Jpeg")

pickle.dump(images, open(os.path.join(pickle_path, "images.pickle"), "wb"))
pickle.dump(coarse_labels, open(os.path.join(pickle_path, "coarse_labels.pickle"), "wb"))
pickle.dump(fine_labels, open(os.path.join(pickle_path, "fine_labels.pickle"), "wb"))

images = pickle.load(open(os.path.join(pickle_path, "images.pickle"), "rb"))
coarse_labels = pickle.load(open(os.path.join(pickle_path, "coarse_labels.pickle"), "rb"))
fine_labels = pickle.load(open(os.path.join(pickle_path, "fine_labels.pickle"), "rb"))
#
# images_confectionary = []
# images_icecream = []
# images_laundry = []
# images_softdrinks_1 = []
# images_softdrinks_2 = []
#
# fine_labels_confectionary = []
# fine_labels_icecream = []
# fine_labels_laundry = []
# fine_labels_softdrinks_1 = []
# fine_labels_softdrinks_2 = []
#
# for image, category, fine_label in zip(images_unpickled, coarse_labels_unpickled, fine_labels_unpickled):
#     if category == 'confectionery':
#         images_confectionary.append(image)
#         fine_labels_confectionary.append(fine_label)
#     elif category == 'icecream':
#         images_icecream.append(image)
#         fine_labels_icecream.append(fine_label)
#     elif category == 'laundry':
#         images_laundry.append(image)
#         fine_labels_laundry.append(fine_label)
#     elif category == 'softdrinks-I':
#         images_softdrinks_1.append(image)
#         fine_labels_softdrinks_1.append(fine_label)
#     elif category == 'softdrinks-II':
#         images_softdrinks_2.append(image)
#         fine_labels_softdrinks_2.append(fine_label)
#
# images_list = [images_confectionary,
#                images_icecream,
#                images_laundry,
#                images_softdrinks_1,
#                images_softdrinks_2]
#
# fine_labels_list = [fine_labels_confectionary,
#                     fine_labels_icecream,
#                     fine_labels_laundry,
#                     fine_labels_softdrinks_1,
#                     fine_labels_softdrinks_2]
#
# categories_list = ['confectionery',
#                    'icecream',
#                    'laundry',
#                    'softdrinks-I',
#                    'softdrinks-II']

hyperparameters = {
    "num_of_grid": [4],
    "interpolated": [False],
    "bins": [30],
    "k": [1],
    "distance_measure": ["l1"]
}
# hyperparameters = {
#     "num_of_grid": [1,2,4,8],
#     "interpolated": [False, True],
#     "bins": [5, 10, 15, 20],
#     "k": [1, 5, 11, 15, 25, 45],
#     "distance_measure": ["l1", "l2"]
# }

# hyperparameters = {
#     "num_of_grid": [1],
#     "interpolated": [True],
#     "bins": [20],
#     "k": [45],
#     "distance_measure": ["l2"]
# }


train_split = 0.8

# for images, fine_labels, category in zip(images_list, fine_labels_list, categories_list):
for feature in ["combined"]:
    results = []
    print(f" Feature: {feature}")
    for num_of_grid in hyperparameters["num_of_grid"]:
        for bins in hyperparameters["bins"]:
            for interpolated in hyperparameters["interpolated"]:
                t0 = time.time()
                gradient_features = []
                color_features = []

                for i, (image, fine_label) in enumerate(zip(images, fine_labels)):
                    grids = grid_images(image, num_of_grid=num_of_grid)
                    gradient_feature = []
                    color_feature = []
                    for grid in grids:
                        gradient_feature.append(
                            get_gradient_orientation_histogram_vector(grid, bins=bins, interpolated=interpolated))
                        color_feature.append(get_color_histogram_vector(grid, bins=bins, interpolated=interpolated))
                    gradient_features.append(np.ravel(gradient_feature))
                    color_features.append(np.ravel(color_feature))

                color_train, color_test, gradient_train, gradient_test, fine_label_train, fine_label_test = train_test_split(
                    color_features,
                    gradient_features,
                    fine_labels,
                    test_size=1 - train_split,
                    stratify=fine_labels)

                print(f"Feature Extraction Time: {time.time() - t0}")
                for k in hyperparameters["k"]:

                    t0 = time.time()
                    for distance_measure in hyperparameters["distance_measure"]:
                        clf = KNN_classifier(color_features=color_train,
                                             gradient_features=gradient_train,
                                             labels=fine_label_train,
                                             k=k,
                                             distance_measure=distance_measure,
                                             feature=feature)

                        score, predictions = clf.score(color_test, gradient_test, fine_label_test)
                        result = {
                            "num_of_grid": num_of_grid,
                            "interpolated": interpolated,
                            "bins": bins,
                            # "k": k,
                            "distance_measure": distance_measure,
                            "feature": feature,
                            "score": score
                        }
                        print(result)
                        results.append(result)

                        with open(f'results_{feature}_features.json', 'w') as fp:
                            json.dump(results, fp)

                        cm = ConfusionMatrix(fine_label_test, predictions)
                        with open(f"pickles/cm_feature_{feature}_grid_{num_of_grid}_bin_{bins}_interpolated_{interpolated}_k_{k}_distance_{distance_measure}.pickle", 'wb') as f:
                            pickle.dump(cm, f)
                        cm.plot(normalized=True)
                        plt.savefig(
                            f"result_feature_{feature}_grid_{num_of_grid}_bin_{bins}_interpolated_{interpolated}_k_{k}_distance_{distance_measure}.png")

                        print(f"KNN calculation time: {time.time() - t0}")
