
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import cv2

class RegistrationByFeatures:


    def findFeatures(self, img):
        """
        Find strong features in the image to use for registration
        :param img: retuna image.
        :return: feature points.
        """

        # using ORB
        # orb = cv2.ORB_create()
        # keypoints, descriptors = orb.detectAndCompute(img, None)

        # using SIFT
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # img2 = cv2.drawKeypoints(img, keypoints, img)
        # cv2.imshow('img', img2) # TODO
        # cv2.waitKey() # TODO

        return keypoints, descriptors

    def calcPointBasedReg(self, FUPoints, BLPoints):
        """
        The function returns a 3x3 rigid 2D transformation matrix of the two translations
        and rotations of the given points and pairings.
        The matrix is such that applying it to FUPoints yields the points that
        are closest (least squared distance) to the BLPoints.
        ------------
        Algorithm is taken from page 5 in this article:
        https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        --------------
        :param BLPoints:
        :param FUPoints:
        :return: rigidReg - a rigid transformation. should use on row vector point.
                            (if want to apply on col vector point: first transpose rigidReg)
        """
        points_num = BLPoints.shape[0]

        # Compute the centroids of both point sets
        centroid_BL = BLPoints.sum(axis=0) / points_num
        centroid_FU = FUPoints.sum(axis=0) / points_num

        # Compute the centered vectors
        centered_BLPoints = BLPoints - centroid_BL
        centered_FUPoints = FUPoints - centroid_FU

        # Compute the d × d covariance matrix
        S = centered_FUPoints.T @ centered_BLPoints

        # Compute SVD of S
        U, sigma, Vt = np.linalg.svd(S)

        # Compute rotation matrix R = VGU^T
        V = Vt.T
        Ut = U.T
        det = np.linalg.det(V @ Ut)

        G = np.zeros((V.shape[1], Ut.shape[0]))
        diagonal = [1] * V.shape[1]
        diagonal[-1] = det
        np.fill_diagonal(G, diagonal)

        R = V @ G @ Ut

        # Compute translation vector
        t = centroid_BL - (R @ centroid_FU)

        # Construct rigidReg = [R(2x2) 0 ; t(1x2) 1]
        # Note I transposed R here, since we work with row vectors while the article
        # works with column vectors

        rigidReg = np.vstack((R.T, t))
        rigidReg = np.concatenate((rigidReg, np.array([[0], [0], [1]])), axis=1)

        return rigidReg

    def ransac(self, x, y, funcFindF, funcDist, minPtNum, iterNum, thDist, thInlrRatio):
        """
        Use RANdom SAmple Consensus to find a fit from X to Y.
        :param x: M*n matrix including n points with dim M
        :param y: N*n matrix including n points with dim N
        :param funcFindF: a function with interface f1 = funcFindF(x1,y1) where:
                    x1 is M*n1
                    y1 is N*n1 n1 >= minPtNum
                    f is an estimated transformation between x1 and y1 - can be of any type
        :param funcDist: a function with interface d = funcDist(x1,y1,f) that uses f returned by funcFindF and returns the
                    distance between <x1 transformed by f> and <y1>. d is 1*n1.
                    For line fitting, it should calculate the distance between the line and the points [x1;y1];
                    For homography, it should project x1 to y2 then calculate the dist between y1 and y2.
        :param minPtNum: the minimum number of points with whom can we find a fit. For line fitting, it's 2. For
                    homography, it's 4.
        :param iterNum: number of iterations (== number of times we draw a random sample from the points
        :param thDist: inlier distance threshold.
        :param thInlrRatio: ROUND(THINLRRATIO*n) is the inlier number threshold
        :return: [f, inlierIdx] where: f is the fit and inlierIdx are the indices of inliers

        transalated from matlab by Adi Szeskin.
        """

        ptNum = len(x)
        thInlr = round(thInlrRatio * ptNum)

        inlrNum = np.zeros([iterNum, 1])
        fLib = np.zeros(shape=(iterNum, 3, 3))

        for i in range(iterNum):
            permut = np.random.permutation(ptNum)
            sampleIdx = permut[range(minPtNum)]
            f1 = funcFindF(x[sampleIdx, :], y[sampleIdx, :])
            dist = funcDist(x, y, f1)
            b = dist <= thDist
            r = np.array(range(len(b)))
            inlier1 = r[b]
            inlrNum[i] = len(inlier1)

            if len(inlier1) < thInlr:
                continue

            fLib[i] = funcFindF(x[inlier1, :], y[inlier1, :])

        idx = inlrNum.tolist().index(max(inlrNum))
        f = fLib[idx]
        dist = funcDist(x, y, f);
        b = dist <= thDist
        r = np.array(range(len(b)))
        inlierIdx = r[b]

        return f, inlierIdx

    def cartesianToHomogeneous(self, Points):
        """
        Convert each point to homogeneous coordinates.
        :param Points: list of row vectors. each vector is a point.
        """
        points_num = Points.shape[0]
        return np.concatenate((Points, np.ones((points_num, 1))), axis=1)

    def calcDist(self, FUPoints, BLPoints, registration_matrix):
        """
        computes the distance of each transformed point from its matching
        point in pixel units.
        :param FUPoints:
        :param BLPoints:
        :param registration_matrix:
        :return:
        """
        homogeneous_BLPoints = self.cartesianToHomogeneous(BLPoints)
        homogeneous_FUPoints = self.cartesianToHomogeneous(FUPoints)

        new_FUPoints = homogeneous_FUPoints @ registration_matrix

        distances = np.sqrt(((homogeneous_BLPoints - new_FUPoints) ** 2).sum(axis=1))

        return distances

    def calcRobustPointBasedReg(self, FUPoints, BLPoints):
        """
        Compute the transformation with unknown outliers in the pairs list.
        :param FUPoints:
        :param BLPoints:
        :return: [f, inlierIdx] where: f is the fit and inlierIdx are the indices of inliers
        """
        f, inlierIdx = self.ransac(x=FUPoints, y=BLPoints, funcFindF=self.calcPointBasedReg, funcDist=self.calcDist,
                                    minPtNum=4, iterNum=200, thDist=20, thInlrRatio=0.1) # 20, 0.1
        return f, inlierIdx

    def applyTransformation(self, img_FU, rigidReg):
        """
        Perform registration.
        In addition, this function computes a new image consisting of the
        transformed FU image overlaid on top of the the BL image
        :param img_FU:
        :param rigidReg:
        """

        # Apply rigidReg.inverse to the FU image
        tform = transform.AffineTransform(rigidReg.T)  # note we transpose since we are working with col vectors now
        tf_img = transform.warp(img_FU, tform.inverse) # tf_img = transform.warp(img_FU, tform.inverse)

        # Why we take the inverse of the transformation:
        # Because we are trying to reconstruct the image after transformation,
        # it is not useful to see where a coordinate from the input image ends up in the output,
        # which is what the transform gives us. Instead, for every pixel (coordinate) in the output image,
        # we want to figure out where in the input image it comes from. Therefore, we need to use the
        # inverse of tform, rather than tform directly.

        return tf_img


    def plot_images(self, img_BL, img_FU, BLPoints, FUPoints, inliersIdx=None):
        """

        :param img_BL:
        :param img_FU:
        :param BLPoints:
        :param FUPoints:
        :param inlierIdx:
        :return:
        """

        f, axarr = plt.subplots(1, 2)

        # show images
        axarr[0].imshow(img_BL, "gray")
        axarr[1].imshow(img_FU, "gray")

        # set titles to subplots
        axarr[0].set_title('BaseLine')
        axarr[1].set_title('Follow Up')

        # plot points
        if inliersIdx is not None:
            points_num = BLPoints.shape[0]
            outliersIdx = [i for i in range(1, points_num) if i not in inliersIdx]
            BL_inliers = BLPoints[inliersIdx]
            BL_outliers = BLPoints[outliersIdx]
            FU_inliers = FUPoints[inliersIdx]
            FU_outliers = FUPoints[outliersIdx]

            l1 = axarr[0].scatter(BL_inliers[:, 0], BL_inliers[:, 1], color="red")
            axarr[1].scatter(FU_inliers[:, 0], FU_inliers[:, 1], color="red")
            l2 = axarr[0].scatter(BL_outliers[:, 0], BL_outliers[:, 1], color="blue")
            axarr[1].scatter(FU_outliers[:, 0], FU_outliers[:, 1], color="blue")

            plt.legend([l1, l2], ["inliers", "outliers"], bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        else:
            axarr[0].scatter(BLPoints[:, 0], BLPoints[:, 1], color="red")
            axarr[1].scatter(FUPoints[:, 0], FUPoints[:, 1], color="red")

        # plot points names
        points_num = BLPoints.shape[0]
        for i in range(0, points_num):
            axarr[0].annotate(i + 1, BLPoints[i], color="orange")
            axarr[1].annotate(i + 1, FUPoints[i], color="orange")

        plt.show()

    def featuresMatching(self, kp1, des1, kp2, des2, img1, img2):
        """

        :param kp1:
        :param des1:
        :param kp2:
        :param des2:
        :return:
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        BLPoints = []
        FUPoints = []
        for m in matches:
            BLPoints.append(list(kp1[m.queryIdx].pt))
            FUPoints.append(list(kp2[m.trainIdx].pt))

        BLPoints = np.array(BLPoints)
        FUPoints = np.array(FUPoints)

        # Display Matches
        img3 = np.zeros(img1.shape)
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=0)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], outImg=img3, **draw_params)
        plt.imshow(img3)
        plt.show()

        return BLPoints, FUPoints, matches

    def featuresMatchingKNN(self, kp1, des1, kp2, des2, img1, img2):

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = np.zeros(img1.shape)
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=0)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches,  outImg=img3, **draw_params)
        plt.imshow(img3)
        plt.show()

        BLPoints = []
        FUPoints = []
        for m in good_matches:
            BLPoints.append(list(kp1[m[0].queryIdx].pt))
            FUPoints.append(list(kp2[m[0].trainIdx].pt))

        BLPoints = np.array(BLPoints)
        FUPoints = np.array(FUPoints)

        return BLPoints, FUPoints, good_matches

    def doRegistration(self, img_BL, img_FU, ROBUST=True):
        """

        :param img_BL: Baseline image
        :param img_FU: Follow-up image
        :param ROBUST: if ROBUST == True, use RANSAC algorithm for finding the best transformation (RANSAC can handle outliers)
                       else, use the close formula for finding the solution
        :return: the transformed img_FU (registrated according to img_BL)
        """

        # Features detecting
        kp1, des1 = self.findFeatures(img_BL)
        kp2, des2 = self.findFeatures(img_FU)

        # Features matching
        #BLPoints, FUPoints, matches = self.featuresMatching(kp1, des1, kp2, des2, img_BL, img_FU)
        BLPoints, FUPoints, matches = self.featuresMatchingKNN(kp1, des1, kp2, des2, img_BL, img_FU)

        # Calculate registration matrix
        if ROBUST == True:
            registration_matrix, inlier_indx = self.calcRobustPointBasedReg(FUPoints, BLPoints)
            self.plot_images(self.__imgBL, self.__imgFU, BLPoints, FUPoints, inlier_indx)
        else:
            registration_matrix = self.calcPointBasedReg(FUPoints, BLPoints)

        # Registration
        tf_img = self.applyTransformation(img_FU, registration_matrix)

        return tf_img