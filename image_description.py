import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_color_histogram(channel, max_value, bins, interpolated=True):
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


def get_gradient_histogram(magnitude, angles, max_value=180, bins=10, interpolated=True):
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


def get_color_histogram_vector(image_rgb, bins=10, interpolated=True):
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
    hist_h = get_color_histogram(image_hsv[:, :, 0], 179, bins=bins, interpolated=interpolated)
    hist_s = get_color_histogram(image_hsv[:, :, 0], 255, bins=bins, interpolated=interpolated)
    hist_v = get_color_histogram(image_hsv[:, :, 0], 255, bins=bins, interpolated=interpolated)
    return np.concatenate((hist_h, hist_s, hist_v))


def get_gradient_x(image):
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)


def get_gradient_y(image):
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)


def get_gradient_magnitude(grad_x, grad_y):
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def get_gradient_angles(grad_x, grad_y):
    return (np.arctan2(grad_y, grad_x) * 180 / np.pi) % 180


def get_gradient_orientation_histogram_vector(image, bins=10, interpolated=True):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = get_gradient_x(image_gray)
    grad_y = get_gradient_y(image_gray)
    magnitude = get_gradient_magnitude(grad_x, grad_y)
    angles = get_gradient_angles(grad_x, grad_y)
    return get_gradient_histogram(magnitude, angles, 180, bins=bins, interpolated=interpolated)
