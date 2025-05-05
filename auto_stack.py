#!/usr/bin/env python

import os
import cv2
import numpy as np
from time import time


# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECC(file_list):
    assert len(file_list) > 0

    M = np.eye(3, 3, dtype=np.float32)

    stacked_image = cv2.imread(file_list[0], 1).astype(np.float32) / 255
    print(file_list[0])
    first_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)

    for file in file_list[1:]:
        image = cv2.imread(file, 1).astype(np.float32) / 255
        print(file)
        # Estimate perspective transform
        _, M = cv2.findTransformECC(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            first_image,
            M,
            cv2.MOTION_HOMOGRAPHY,
        )
        w, h, _ = image.shape
        # Align image to first image
        image = cv2.warpPerspective(image, M, (h, w))
        stacked_image += image

    stacked_image /= len(file_list)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(file_list):
    assert len(file_list) > 0

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    print(file_list[0])
    first_image = cv2.imread(file_list[0], 1)
    stacked_image = first_image.astype(np.float32) / 255
    # compute the descriptors with ORB
    kp = orb.detect(first_image, None)
    first_kp, first_des = orb.compute(first_image, kp)

    for file in file_list[1:]:
        print(file)
        image = cv2.imread(file, 1)
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Find matches and sort them in the order of their distance
        matches = matcher.match(first_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate perspective transformation
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        w, h, _ = imageF.shape
        imageF = cv2.warpPerspective(imageF, M, (h, w))
        stacked_image += imageF

    stacked_image /= len(file_list)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image


# ===== MAIN =====
# Read all files in directory
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_dir", help="Input directory of images ()")
    parser.add_argument("output_image", help="Output image name")
    parser.add_argument(
        "--method", help="Stacking method ORB (faster) or ECC (more precise)"
    )
    parser.add_argument("--show", help="Show result image", action="store_true")
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print("ERROR {} not found!".format(image_folder))
        exit()

    file_list = os.listdir(image_folder)
    file_list = [
        os.path.join(image_folder, x)
        for x in file_list
        if x.endswith((".jpg", ".png", ".bmp"))
    ]

    if args.method is not None:
        method = str(args.method)
    else:
        method = "KP"

    tic = time()

    if method == "ECC":
        # Stack images using ECC method
        description = "Stacking images using ECC method"
        print(description)
        stacked_image = stackImagesECC(file_list)

    elif method == "ORB":
        # Stack images using ORB keypoint method
        description = "Stacking images using ORB method"
        print(description)
        stacked_image = stackImagesKeypointMatching(file_list)

    else:
        print("ERROR: method {} not found!".format(method))
        exit()

    print("Stacked {0} in {1} seconds".format(len(file_list), (time() - tic)))

    print("Saved {}".format(args.output_image))
    cv2.imwrite(str(args.output_image), stacked_image)

    # Show image
    if args.show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
