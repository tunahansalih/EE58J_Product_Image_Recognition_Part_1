import cv2
import numpy as np
import matplotlib.pyplot as plt


def createHistogramFromMagnitudeAndOrientation(magnitude, orientation, max_value, bins):
    """
    :param magnitude: numpy nd-array containing magnitude of cells
    :param orientation: numpy nd-array containing orientation of cells
    :param bins: integer, number of bins to calculate
    :return: numpy nd-array, containing histogram of oriented gradients for given cell
    """
    bin_size = max_value / bins
    cell_histogram = np.zeros(bins)
    for y in range(orientation.shape[0]):
        for x in range(orientation.shape[1]):
            current_orientation = orientation[y, x]
            current_magnitude = magnitude[y, x]
            # Interpolate magnitude over neighbor bins
            former_bin = int(current_orientation / bin_size)
            latter_bin = former_bin + 1
            weight_of_former_bin = 1 + former_bin - current_orientation / bin_size
            weight_of_latter_bin = 1 - weight_of_former_bin

            cell_histogram[former_bin % bins] += current_magnitude * weight_of_former_bin
            cell_histogram[latter_bin % bins] += current_magnitude * weight_of_latter_bin
    return cell_histogram


def create_color_histogram(channel, max_value, bins, interpolated=True):
    """

    :param channel: the color channel to be discretized as a histogram
    :param max_value: the maximum value that the channel can have
    :param bins: the number of equal-width bins
    :param interpolated: boolean value that defines whether interpolation between neighbor values will be done
    :return:
    """
    if interpolated:
        bin_size = max_value / bins
        cell_histogram = np.zeros(bins, dtype=np.float)
        values = np.ravel(channel)
        former_bins = np.array(values / bin_size - 0.5, dtype=np.int)
        latter_bins = former_bins + 1
        weights_of_former_bins = (1 + (former_bins + 0.5) - values / bin_size) % (1.0 + 1e-15)
        weights_of_latter_bins = 1 - weights_of_former_bins
        former_indices = np.array(former_bins % bins, dtype=np.int)
        latter_indices = np.array(latter_bins % bins, dtype=np.int)
        np.add.at(cell_histogram, former_indices, weights_of_former_bins)
        np.add.at(cell_histogram, latter_indices, weights_of_latter_bins)
        cell_histogram /= np.sum(np.abs(cell_histogram))
        return cell_histogram
    else:
        bin_size = max_value / bins
        values = np.ravel(channel)
        indices = np.clip(values // bin_size, 0, bins - 1).astype(np.int)
        cell_histogram = np.zeros(bins, dtype=np.float)
        np.add.at(cell_histogram, indices, 1)
        cell_histogram /= np.sum(np.abs(cell_histogram))
        return cell_histogram


def create_gradient_histogram(magnitude, angles, max_value, bins, interpolated=True):
    if interpolated:
        bin_size = max_value / bins
        cell_histogram = np.zeros(bins, dtype=np.float)
        angle_values = np.ravel(angles)
        magnitude_values = np.ravel(magnitude)
        former_bins = np.array(angle_values / bin_size - 0.5, dtype=np.int)
        latter_bins = former_bins + 1
        weights_of_former_bins = (1 + (former_bins + 0.5) - angle_values / bin_size) % (1.0 + 1e-15)
        weights_of_latter_bins = 1 - weights_of_former_bins
        former_indices = np.array(former_bins % bins, dtype=np.int)
        latter_indices = np.array(latter_bins % bins, dtype=np.int)
        np.add.at(cell_histogram, former_indices, weights_of_former_bins * magnitude_values)
        np.add.at(cell_histogram, latter_indices, weights_of_latter_bins * magnitude_values)
        cell_histogram /= np.sum(np.abs(cell_histogram))
        return cell_histogram
    else:
        bin_size = max_value / bins
        angle_values = np.ravel(angles)
        magnitude_values = np.ravel(magnitude)
        indices = np.clip(angle_values // bin_size, 0, bins - 1).astype(np.int)
        cell_histogram = np.zeros(bins, dtype=np.float)
        np.add.at(cell_histogram, indices, magnitude_values)
        cell_histogram /= np.sum(np.abs(cell_histogram))
        return cell_histogram


def grid_images(image, num_of_grid):
    image_windows = []
    window_size = (image.shape[0] // num_of_grid)
    for grid_y in range(num_of_grid):
        for grid_x in range(num_of_grid):
            image_windows.append(
                image[grid_y * window_size:(grid_y + 1) * window_size, grid_x * window_size:(grid_x + 1) * window_size])
    return image_windows


image = cv2.imread(
    "/home/tunahansalih/PycharmProjects/EE58J_Assignment_1/data/SKU_Recognition_Dataset/confectionery/6753/crop_5744388.jpg")
image = cv2.resize(image, (128, 128))
grids = grid_images(image, 4)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist_h = create_color_histogram(image_hsv[:, :, 0], 179, 10)
hist_s = create_color_histogram(image_hsv[:, :, 0], 255, 10)
hist_v = create_color_histogram(image_hsv[:, :, 0], 255, 10)

# hist = create_color_histogram_interpolated([[[179]]], 179, 10)
