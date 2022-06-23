from registration_by_features import RegistrationByFeatures
from registration_by_intensity import RegistrationByIntensity

import matplotlib.pyplot as plt
import sys
import cv2

ARGS_NUM = 2

def rgb2gray(rgb):
    """
    Convert the given RGB image to a graysclae image.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def display_result(img_BL, tf_img):
    """

    :param img_BL: Baseline image
    :param tf_img: FU transformed image
    """
    # both images (overlaid)
    plt.figure(figsize=(10, 10))
    plt.title('BL & transformed FU', fontsize=25)
    plt.imshow(img_BL, "gray")
    plt.imshow(tf_img, alpha=0.4)
    plt.show()

    # BL image
    plt.figure(figsize=(10, 10))
    plt.title('BL', fontsize=25)
    plt.imshow(img_BL, "gray")
    plt.show()

    # transformed image
    plt.figure(figsize=(10, 10))
    plt.title('transformed FU', fontsize=25)
    plt.imshow(tf_img, "gray")
    plt.show()


if __name__ == '__main__':

    if len(sys.argv)-1 != ARGS_NUM:
        raise Exception("Enter the paths to the Baseline and Follow-up images")

    # Load images as grayscale images

    img_BL_path = sys.argv[1]
    img_FU_path = sys.argv[2]

    img_BL = cv2.imread(img_BL_path, cv2.IMREAD_GRAYSCALE)
    img_FU = cv2.imread(img_FU_path, cv2.IMREAD_GRAYSCALE)

    if img_BL is None or img_FU is None:
        raise Exception("Please verify that the given paths are valid")

    # First Algorithm

    regByFeatures = RegistrationByFeatures()
    transformed_FU = regByFeatures.doRegistration(img_BL, img_FU)
    display_result(img_BL, transformed_FU)

    # Second Algorithm

    regByIntensity = RegistrationByIntensity()
    transformed_FU = regByIntensity.doRegistration(img_BL, img_FU)
    display_result(img_BL, transformed_FU)
