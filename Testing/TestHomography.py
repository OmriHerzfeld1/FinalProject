from __future__ import print_function
import cv2
import numpy as np
# import SimpleITK as sitk
import os

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.2


def align2(im1, im2):
    # Convert to grayscale.
    img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(MAX_FEATURES)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Take the top GOOD_MATCH_PERCENT % matches forward.
    matches = matches[:int(len(matches) * GOOD_MATCH_PERCENT)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    print(homography)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(im1,
                                          homography, (width, height))

    # Save the output.
    cv2.imwrite('output.jpg', transformed_img)

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(matcher.match(descriptors1, descriptors2, None))


    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    ransac_reproj_threshold = 2
    h, mask = cv2.estimateAffine2D(points1, points2, ransacReprojThreshold=ransac_reproj_threshold)
    h = np.vstack([h, [0, 0, 1]])
    print(h)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def cal_error(im1, im2):
    # Compute the pixel-wise difference between the images
    diff = im1.astype(np.float32) - im2.astype(np.float32)

    # Compute the squared error
    squared_error = np.square(diff)

    # Compute the sum of squared error
    sum_squared_error = np.sum(squared_error)

    # Compute the mean squared error
    mean_squared_error = sum_squared_error / (im1.shape[0] * im1.shape[1])

    # Compute the root mean squared error (L2 error)
    l2_error = np.sqrt(mean_squared_error)

    return l2_error


if __name__ == '__main__':

    # Read reference image
    refFilename = "origin.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "homography.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    ## cv image registration
    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # checking the error
    error = cal_error(imReference, imReg)
    print(f"The error of the trans: {error}")
    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # testing align
    # align2(im, imReference)
    # testing sitk
    # TestSimpleITK(imReference, im)
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # shit=np.array([[0,1,0], [1,0,0], [0,0,1]],dtype='float')
    # # ggg = cv2.warpPerspective(gray, h, (gray.shape[0], gray.shape[1]))
    # # cv2.imshow("aligned", ggg)
    # cv2.waitKey()
    # #cv2.imwrite("trans", cv2.warpPerspective(im, h, (300, 300)))

    # Print estimated homography
    # print("Estimated homography : \n", h)
