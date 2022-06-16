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

<a href="url"><img src=(http://url.to/image.png" align="left" height="48" width="48](https://user-images.githubusercontent.com/68702877/174074532-bb739f79-4683-4edd-ae1f-fcb19ecbecb7.png) ></a>

![image](https://user-images.githubusercontent.com/68702877/174074532-bb739f79-4683-4edd-ae1f-fcb19ecbecb7.png)


![image](https://user-images.githubusercontent.com/68702877/174075445-96323638-dc92-44ae-8797-baed2aec0a6e.png)


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

Algorithm:
1.	Features Detecting - I used Sift.
2.	Features Matching - I used KNN matching.
3.	Pick matched points - I picked matches (m1, m2) that pass the ratio test: (m1.distance / m2.distance) < 0.75. 
4.	Registration - I calculate the registration matrix using the picked matches. The solution is based on SVD. See page 5 in this article: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf


![image](https://user-images.githubusercontent.com/68702877/174043592-1e24971b-9372-45d7-9dbd-ee24d6315115.png)


![image](https://user-images.githubusercontent.com/68702877/174043318-035b3866-604e-44e2-84d0-2f09e7945f8b.png)

Matches after steps 1-3. Red dots that has no  green line which connects between them are outliers matches.

## Intensity-based Registration

Algorithm:
0.	Get BL and FU retina blood vessels segmentaions.
1.	For each angle in [-30, 30]:
  •	Rotate FU segmentation in that angle
  •	Perform cross-correlation between rotated FU segmentation and BL segmentation, and save the result in a list. 
2.	Find the translation vector & angle which provided the minimum error.


**Segment Retinal Blood Vessels**
In this section I implemented an algorithm which gets an image of human retina, and oupts a segmentation of the blood vessels in the retina.
Algorithm :
0. Convert image from RGB to grayscale image.
1. Contrast Enhancement – using CLAHE
2. Background Exclusion
3. Thresholding
4. Morphological operations

1.	Contrast Enhancement – using CLAHE
In this step I apply on the image a type of histogram equalization named CLAHE (contrast-limited adaptive histogram equalization), in order to deepen the contrast of the image.
2.	Background Exclusion
In this step I subtract from the image (from step 1) its blurred  image, in order to eliminating background variations in illumination such that the foreground objects (in our case the blood vessels) may be more easily analyzed.
3.	Thresholding
In this step I perform thresholding using isodata in order to select the threshold value. Note that before I do that, I blur the image.
4.	Morphological operations
In this step I perform some morphological operations (such as opening and closing) in order to discard noise.



|  |  |  |
| ---  | ---  | --- |
| **A** – Original image | **B** – Contrast Enhancement | **C** – Background Exclusion |
|![image](https://user-images.githubusercontent.com/68702877/174040994-d87c15c3-40c8-4177-8f7f-bb30ac8f6e16.png)|![image](https://user-images.githubusercontent.com/68702877/174041064-673d5bac-d72a-4d62-8b1a-8bdf77109c2c.png)|![image](https://user-images.githubusercontent.com/68702877/174041081-0e2baa16-dc79-4d20-a742-2462e3f57e4e.png)|
| **D** – Thresholding | **E** – Opening (3,3) | **F** – Opening (5,5) |
|![image](https://user-images.githubusercontent.com/68702877/174041307-9f68cced-cb15-4fb8-999e-a859782e451e.png)|![image](https://user-images.githubusercontent.com/68702877/174041338-0b7de23c-90d3-4a4a-870a-3149c1538c6f.png)|![image](https://user-images.githubusercontent.com/68702877/174041353-a6a398d8-01fa-44b6-a772-46cf6b0e77fd.png)|
| **G** – Opening (7,7) | **H** – Closing (17,17) | **I** – Final segmentation (after remove_small_objects) |
|![image](https://user-images.githubusercontent.com/68702877/174041452-74005e24-5127-4ae1-9c6c-a1ded06ffac3.png)|![image](https://user-images.githubusercontent.com/68702877/174041478-25a51513-c6a3-4009-9a50-88d0227a5001.png)|![image](https://user-images.githubusercontent.com/68702877/174041488-437caf62-086b-4345-8eb3-e2f6ac85b048.png)|



![image](https://user-images.githubusercontent.com/68702877/174043801-bec60e1f-e53b-4ef6-8bb7-31d3edfb7455.png)

