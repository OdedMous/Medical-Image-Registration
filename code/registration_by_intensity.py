import numpy as np
from skimage.filters import threshold_isodata
from skimage import morphology
import cv2
from scipy import ndimage
from skimage.registration import phase_cross_correlation

class RegistrationByIntensity:


    def SegmentBloodVessel(self, img):
        """
        Perform segmentation of the blood vessels in the retina.
        :param img: retina image.
        :return: blood vessels segmentation.
        """

        # Contrast Enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(img)
        # show_image(cl_img) # TODO

        # Background Exclusion
        blur = cv2.GaussianBlur(cl_img, (57, 57), 0)
        front_img = np.clip(cl_img - blur, 0, 255)
        # show_image(front_img)  # TODO

        # Thresholding
        front_img = cv2.GaussianBlur(front_img, (3, 3), 0)

        thresh = threshold_isodata(front_img)

        segmentation = np.zeros(img.shape)
        low_values_flags = front_img < thresh
        high_values_flags = front_img >= thresh
        segmentation[low_values_flags] = 0
        segmentation[high_values_flags] = 1
        # show_image(segmentation)  # TODO

        # Opening
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(segmentation, cv2.MORPH_OPEN, kernel)
        # show_image(opening)  # TODO

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        # show_image(opening)  # TODO

        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        # show_image(opening)  # TODO

        # Closing
        kernel = np.ones((17, 17), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # show_image(closing)  # TODO

        # Remove small objects
        cleaned = morphology.remove_small_objects(closing.astype(bool), min_size=200, connectivity=2)
        # show_image(cleaned)  # TODO

        return cleaned

    def doRegistration(self, img_BL, img_FU):

        # Perform retina segmentaion on both images
        seg_BL = self.SegmentBloodVessel(img_BL)
        seg_FU = self.SegmentBloodVessel(img_FU)

        shifts_lists = []
        for angle in np.arange(-30, 30, 1): # TODO -30 30

            # Rotate
            rotated_FU = ndimage.rotate(seg_FU, angle, reshape=False)

            #  phase_cross_correlation
            detected_shift = phase_cross_correlation(seg_BL, rotated_FU) # shift, error, diffphase
            shift, error, diffphase = detected_shift
            shifts_lists.append((shift, error, diffphase, angle))

        # Find the min error in that list
        optimal_translation_and_rotation = min(shifts_lists, key=lambda t: t[1])
        optimal_translation = np.flip(optimal_translation_and_rotation[0])
        optimal_angle = optimal_translation_and_rotation[3]
        print("optimal angle:", optimal_angle)


        # Apply translation & rotate to FU image
        T_mat = np.float32([[1, 0, optimal_translation[0]], [0, 1, optimal_translation[1]]])
        height, width = img_FU.shape[:2]
        img_translation = cv2.warpAffine(img_FU, T_mat, (width, height))
        rotated_translation = ndimage.rotate(img_translation, optimal_angle, reshape=False)

        return rotated_translation