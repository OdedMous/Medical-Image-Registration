# Medical-Image-Registration

| Base-Line image | Follow-Up image (registrated) | Combined |
| ---  | ---  |  :----:  |
| ![image](https://user-images.githubusercontent.com/68702877/174038357-fed32ffb-1a0a-4f00-b44a-aec64006e09e.png) |![image](https://user-images.githubusercontent.com/68702877/174038420-b8e9257d-a375-411a-b17e-7bcdc59759af.png)| ![image](https://user-images.githubusercontent.com/68702877/174038468-a8d47cb2-98fd-44a1-bd38-a3a6dad5f5ab.png)|


## Goal
The goal of this project is to implement automatic registration algorithms between two retinal 2D scans. Two different techniques are implemented: geometry-based registration and intensity-based registration.

## Background

**Registration in Medicine**

Usually, there is significant movement between two images of the same patient taken at two different times. This is because the patient is in different poses, because of internal movements (e.g., breathing) and because of other physical changes that occurred in the time that passed between the scans. **Registering** the images allows one to perform a comparison between them, e.g. to track the differences, or to evaluate the efficacy of a treatment when baseline and follow-up images are provided.

In this project we used pairs of **retinal 2D scans** of patients who suffered from wet age-related macular degeneration (wet AMD), an eye disease that causes blurred vision or a blind spot in the visual field. It's generally caused by abnormal blood vessels that leak fluid or blood into the macula.
The first scan in each pair is a baseline image and the second is an image that was taken later on in time in order to examine how the disease has evolved.

| AMD condition | Example of blind spot |
| ---  | ---  | 
|![e](https://user-images.githubusercontent.com/68702877/174078749-74593be7-3ffb-439d-a255-825a6fd989d5.png)|![image](https://user-images.githubusercontent.com/68702877/174075445-96323638-dc92-44ae-8797-baed2aec0a6e.png)|


**Rigid Registration**

In this project we assume that the anatomical structures of interest in the images retain their shape, and hence a rigid transform is sufficient to align the two images. 
Rigid registration consists of computing the translation and rotation. Rigid registration of 2D images requires computing **three
parameters**: two translations and one rotation.

![tempFileForShare_20220616-155509](https://user-images.githubusercontent.com/68702877/174074392-38df3481-a57e-4b2f-855e-714e047b65de.jpg)


Rigid registration algorithms can be categorized into two groups: **geometry** and **intensity based**. In geometric-based
algorithms, the features to be matched, e.g. points, are first identified in each image and then paired. The sum of the
square distances between the points is then minimized to find the rigid transformation. In intensity-based algorithms,
a similarity measure is defined between the images. The transformation that maximizes the similarity is the desired
one. The rigid registration algorithms are iterative – in each step, a transformation is generated and tested to see if it
reduces the sum of the squared distances between the points or increases the similarity between the images.


## Geometry-based Registration

**Algorithm:**
1.	Features Detecting - using SIFT algorithm.
2.	Features Matching - using KNN matching.
3.	Homography Computation - The registration matrix is calculated using the picked matches. RANSAC algorithm is used in order to handle outlier matches. The RANSAC algorithm is implemented using a solution which based on SVD, see page 5 in this article: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

![image](https://user-images.githubusercontent.com/68702877/174043318-035b3866-604e-44e2-84d0-2f09e7945f8b.png)

Matches after steps 1-3. Red dots that has no  green line which connects between them are outliers matches.

## Intensity-based Registration

**Algorithm:** </br>
1.	Get BL and FU retina blood vessels segmentaions.
2.	For each angle in [-30, 30]: </br>
  •	Rotate FU segmentation in that angle </br>
  •	Perform cross-correlation between rotated FU segmentation and BL segmentation, and save the result in a list. 
3.	Find the translation vector & angle which provided the minimum error.


**Segment Retinal Blood Vessels** </br>
In this section I implemented an algorithm which gets an image of human retina, and oupts a segmentation of the blood vessels in the retina. </br>

**Algorithm:** </br>
1. Convert image from RGB to grayscale. 
2.	Contrast Enhancement – In this step a type of histogram equalization named CLAHE (contrast-limited adaptive histogram equalization) is applied, in order to deepen the contrast of the image.
3.	Background Exclusion - In this step I subtract from the image (from the previous step) its blurred  image, in order to eliminating background variations in illumination such that the foreground objects (in our case the blood vessels) may be more easily analyzed.
4.	Thresholding - In this step thresholding is applied using isodata algorithm.
5.	Morphological operations - In this step some morphological operations are performed (such as opening and closing) in order to discard noise.

|  |  |  |
| ---  | ---  | --- |
| **A** – Original image | **B** – Contrast Enhancement | **C** – Background Exclusion |
|![image](https://user-images.githubusercontent.com/68702877/174040994-d87c15c3-40c8-4177-8f7f-bb30ac8f6e16.png)|![image](https://user-images.githubusercontent.com/68702877/174041064-673d5bac-d72a-4d62-8b1a-8bdf77109c2c.png)|![image](https://user-images.githubusercontent.com/68702877/174041081-0e2baa16-dc79-4d20-a742-2462e3f57e4e.png)|
| **D** – Thresholding | **E** – Opening (3,3) | **F** – Opening (5,5) |
|![image](https://user-images.githubusercontent.com/68702877/174041307-9f68cced-cb15-4fb8-999e-a859782e451e.png)|![image](https://user-images.githubusercontent.com/68702877/174041338-0b7de23c-90d3-4a4a-870a-3149c1538c6f.png)|![image](https://user-images.githubusercontent.com/68702877/174041353-a6a398d8-01fa-44b6-a772-46cf6b0e77fd.png)|
| **G** – Opening (7,7) | **H** – Closing (17,17) | **I** – Final segmentation (after remove_small_objects) |
|![image](https://user-images.githubusercontent.com/68702877/174041452-74005e24-5127-4ae1-9c6c-a1ded06ffac3.png)|![image](https://user-images.githubusercontent.com/68702877/174041478-25a51513-c6a3-4009-9a50-88d0227a5001.png)|![image](https://user-images.githubusercontent.com/68702877/174041488-437caf62-086b-4345-8eb3-e2f6ac85b048.png)|




