from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2 as cv

import torch
import torchvision.transforms as T

from skimage import transform, util, filters, exposure




class Transformations:
    """Transformation functions """

    @staticmethod
    def random_rotation(img: np.ndarray, degree: [float, int]):
        """
        :param img: np.ndarray
        :param degree: float | int
        :return: transformed image
        """
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-degree, degree)
        return transform.rotate(img, random_degree)

    @staticmethod
    def random_noise(img: np.ndarray, noiseType: str, noiseMax: [None, int]):
        """
        add random noise to the image
        :param img: np.ndarray
        :param noiseType:str
        :param noiseMax:str
        :return: transformed image
        """
        return util.random_noise(img, noiseType, noiseMax)

    @staticmethod
    def horizontal_flip(img: np.ndarray):
        """
        horizontal flip doesn't need skimage, flipping the image array of pixels !
        :param img: np.ndarray
        :return: transformed image
        """
        return img[0][::-1, :]

    @staticmethod
    def vertical_flip(img: np.ndarray):
        # vertical flip doesn't need skimage,flipping the image array of pixels !
        return img[0][:, ::-1]

    @staticmethod
    def gauss(img: np.ndarray, sigma:float):
        """
            gauss: standard deviation for Gaussian kernel
            :param img: np.ndarray
            :param  sigma:float
            :return: transformed image
        """
        return filters.gaussian(img, sigma=sigma)

    @staticmethod
    def change_contrast_bright(indata: list):
        """
            lower gama transform
            :param indata: [image, Min, Max]
            :return: transformed image
        """
        gain = 1
        random_gamma = random.uniform(indata[1], indata[2])
        return exposure.adjust_gamma(indata[0], random_gamma, gain)

    @staticmethod
    def change_contrast_dark(img: np.ndarray, gama_min:[int,float], gama_max:[int, float]):
        """
            Higher gama transform
            :param img: [np.ndarray]
            :param gama_min: [int,float]
            :param gama_max: [int,float]

            :return: transformed image
        """
        gain = 1
        random_gamma = random.uniform(gama_min, gama_max)
        return exposure.adjust_gamma(img, random_gamma, gain)

    @staticmethod
    def elastic_distortion(img: np.ndarray):
        # Input needed
        sigma = 5  # Elastic Distortion transformation Standard deviation of Gaussian convolution
        alpha = 5  # Elastic Distortion transformation Scaling factor
        # Compute a random displacement field
        dx = 2 * np.random.rand(img.shape[0], img.shape[1])  # dx ~ U(-1,1)
        dy = 2 * np.random.rand(img.shape[0], img.shape[1])  # dy ~ U(-1,1)
        # Normalizing the field
        nx = np.linalg.norm(dx)
        ny = np.linalg.norm(dy)
        dx = dx / nx  # Normalization: norm(dx) = 1
        dy = dy / ny  # Normalization: norm(dy) = 1
        # Smoothing the field
        fdx = filters.gaussian(dx, sigma=sigma)  # 2-D Gaussian filtering of dx
        fdy = filters.gaussian(dy, sigma=sigma)  # 2-D Gaussian filtering of dy
        # Filter size: 2 * 3*ceil(std2(dx)) + 1
        # = 3 sigma pixels in each direction + 1 to make an odd integer
        fdx = alpha * fdx  # Scaling the filtered field
        fdy = alpha * fdy  # Scaling the filtered field

        # The resulting displacement


if __name__ == "__main__":
    image = cv.imread(r'C:\BM\Data\Aligned\images\SIM_000.jpg')
    img_trc = torch.Tensor(image)
    t_affine = T.RandomAffine(degrees=20, translate=(0.2, 0.5), scale=(1.5, 1.5))
    t1 = T.ColorJitter(brightness=.5)
    t1(img_trc)
    plt.show()
