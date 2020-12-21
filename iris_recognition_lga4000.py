import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from collections import Counter


# wrote a function to read an image and extract the iris out of the image
def hough_transform(filename):
    im = cv2.imread(filename)
    # print(im.shape)
    gray_blurred = cv2.blur(im, (3, 3))
    edges = cv2.Canny(gray_blurred, 10, 50)

    outer_circle_values = []
    inner_circle_values = []

    minRadius = 100
    maxRadius = 140

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 40,
                               param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)

    circles = np.uint16(np.around(circles))

    outer_a, outer_b, outer_r = circles[0][0][0], circles[0][0][1], circles[0][0][2]

    for j in circles[0, :]:
        # draw the outer circle
        cv2.circle(im, (j[0], j[1]), j[2], (0, 255, 0), 2)

       
        # draw the center of the circle
        cv2.circle(im, (j[0], j[1]), 2, (0, 0, 255), 3)
        outer_circle_values.append(j)
        break

    threshold = 20

    inner_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 0.0001, 20,
                                     param1=60, param2=30, minRadius=20, maxRadius=80)
    inner_circles = np.float32(np.around(inner_circles))

    outer_circle_center = np.array(outer_circle_values[0]).astype('float32')

    inner_true_center = []

    flag = 0
    while flag == 0:
        for j in inner_circles[0, :]:
            if np.abs(outer_circle_center[0] - j[0]) < threshold and np.abs(outer_b - j[1]) < threshold:
                inner_true_center.append(j)
                flag = 1

        if len(inner_true_center) == 0:
            threshold += 10

    inner_true_center = np.array(inner_true_center).astype('uint16')
    outer_circle_center = np.array(outer_circle_center).astype('uint16')

    return outer_a, outer_b, outer_r, inner_true_center[0][0], inner_true_center[0][1], inner_true_center[0][2]


# changing the extracted iris into a rectangular shape image
def rectangle_image(outer_a, outer_b, outer_r, inner_a, inner_b, inner_r, filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    center_avg = ((outer_a + inner_a) / 2, (outer_b + inner_b) / 2)

    nsamples = 360

    samples = np.linspace(0, 2 * np.pi, nsamples)[:-1]

    if inner_r > outer_r:
        t = outer_r
        outer_r = inner_r
        inner_r = t

    polar = np.zeros((outer_r - inner_r, nsamples))

    for r in range(inner_r, outer_r):
        for theta in samples:
            x = r * np.cos(theta) + center_avg[0]
            y = r * np.sin(theta) + center_avg[1]

            th = int(theta * nsamples / 2.0 / np.pi)

            polar[r-inner_r][th] = image[int(y)][int(x)]

    return polar


# performing hamming distance to measure the recognition system
def GetHamming_dist(X,y):
    hamming_code = []
    X = np.array(X).astype('int')
    y = np.array(y).astype('int')
    #print(X.shape)
    #print(y.shape)
    if X.shape[0] < y.shape[0]:
        padding = np.zeros((y.shape[0]-X.shape[0],y.shape[1])).astype('int')
        X = np.concatenate((X,padding))
    
    else:
        padding = np.zeros((X.shape[0]-y.shape[0],y.shape[1])).astype('int')
        y = np.concatenate((y,padding))
        
    for i in range(0,X.shape[0]):
        result = 0
        sum_exor = 0
        temp1 = X[i]
        temp2 = y[i]
        
        for i,j in zip(temp1,temp2):
            #print(i,j)
            result = i ^ j
            sum_exor+=result
        hamming_code.append(sum_exor/X.shape[1])
    
    
    return hamming_code
        

def To_Binary(img):
    bin_image = np.ones((img.shape))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            bin_image[i][j] = img[i][j] % 2
            
    
    return bin_image


if __name__ == "__main__":
# "Probe_LG2200_2010/02463d2771.tiff"
    probe_files = os.listdir("Probe_LG4000")
    path = "Probe_LG4000/"
    gallery_files = os.listdir("Gallery/")
    path1 = "Gallery/"

    print("files: ", len(probe_files))
    distribution = np.zeros((len(probe_files), len(gallery_files)))
    
    count_p = 0


    # each probe file
    for i in probe_files:
        # print("in probe file:",i)
        temp = str.split(i, ".")
        if temp[1] != "txt":
            filename_probe = os.path.join(path, i)
            try:
                print("in probe filename:", filename_probe)
                
                #Get the outer and inner circle parameters
                outer_a, outer_b, outer_r, inner_a, inner_b, inner_r = hough_transform(filename_probe)
                # print("got inner outer params: ", outer_a, outer_b, outer_r, inner_a, inner_b, inner_r)
    
                #Convert them into a rectanglar image
                polar = rectangle_image(outer_a, outer_b, outer_r, inner_a, inner_b, inner_r, filename_probe)
    
                #Use gabor filter to extract features
                filters = []
                ksize = 31
                for theta in np.arange(0, np.pi, np.pi / 16):
                    kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                    kern /= 1.5 * kern.sum()
                    filters.append(kern)
    
                accum = np.zeros_like(polar)
                for kern in filters:
                    fimg = cv2.filter2D(polar, cv2.CV_8UC3, kern)
                    np.maximum(accum, fimg, accum)
    
                #Converting to Binary_image
                feature_vector1 = To_Binary(accum)
                # print("got probe feature vector: \n", feature_vector1)
    
                #feature_vectors1.append(feature_vector1)
                #go to each image in gallery
    
    
                count_g = 0
            except:
                print("Next file")
                
            try:
                for j in gallery_files:
                    # print("in galley file: ", j)
                    temp = str.split(j, ".")
                    if temp[1] != "txt":
                        filename_gallery = os.path.join(path1, j)
                        # try:
    
    
                        print("in gallery filename:", filename_gallery)
    
                        #Get the outer and inner circle parameter
                        outer_a, outer_b, outer_r, inner_a, inner_b, inner_r = hough_transform(filename_gallery)
                        # print("got inner outer params in gallery: ", outer_a, outer_b, outer_r, inner_a, inner_b, inner_r)
    
                        polar = rectangle_image(outer_a, outer_b, outer_r, inner_a, inner_b, inner_r, filename_gallery)
    
                        #use the gabor filter to extract features
                        filters = []
                        ksize = 31
                        for theta in np.arange(0, np.pi, np.pi / 16):
                            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                            kern /= 1.5 * kern.sum()
                            filters.append(kern)
    
                        accum = np.zeros_like(polar)
                        for kern in filters:
                            fimg = cv2.filter2D(polar, cv2.CV_8UC3, kern)
                            np.maximum(accum, fimg, accum)
    
                        # Converting to Binary_image
                        feature_vector2 = To_Binary(accum)
                        # print("got gallery feature vector: ", feature_vector2)
                        # save the binay feature vector in a list
                        #feature_vectors2.append(feature_vector2)
                        #calculate the hamming distance between each probe and gallery
    
                        hamming_dist1 = GetHamming_dist(feature_vector1, feature_vector2)
                        distribution[count_p][count_g] = np.mean(hamming_dist1)
                        print("count_p: ", count_p)
                        print("count_g: ", count_g)
                        print("distribution: ",distribution[count_p][count_g] )
                        count_g += 1
    
                count_p += 1
            except:
                print("passed to next file")
            # except:
            #     print("passed to next file")

  
    print(distribution)

    
    c = Counter()
    
    for i in range(distribution.shape[0]):
        for j in range(distribution.shape[1]):
            if distribution[i][j] != 0:
                print(distribution[i][j])
                c[distribution[i][j]]+=1

    # plt.figure(figsize=(100, 100))
    # ypos = np.arange(len(c.keys()))
    plt.bar(c.keys(), c.values(), align='edge', width=0.02)
    plt.grid()
    plt.title("Imposter vs Genuine ")
    plt.ylabel("Frequency")
    plt.xlabel("Hamming distance")
    plt.show()
    

    p = [0.50, 0.4988, 0.497, 0.496,0.495,0.494]
    false_match = []
    true_match_dist = []
    false_nonmatch_dist = []
    true_nonmatch_dist = []
    print("In ROC CURVE")
    for x in p:
        true_match = []
        auth_dist = []
        imposter_dist = []
        false_nonmatch = []
        true_nonmatch = []

        
    # Get  Authentic and imposter distribution using threshold values
    
        for i in range(0, distribution.shape[0]):
            for j in range(0, distribution.shape[1]):
                if distribution[i][j] < x:
                    auth_dist.append(distribution[i][j])

                else:
                    imposter_dist.append(distribution[i][j])

    # Get True match distribution
        for i in range(0, distribution.shape[0]):
            for j in range(0, distribution.shape[1]):
                if i == j and distribution[i][j] < x:
                    true_match.append(distribution[i][j])

    # Get False non match predictions for given threshold
        for i in range(0, distribution.shape[0]):
            for j in range(0, distribution.shape[1]):
                if i == j and distribution[i][j] > x:
                    false_nonmatch.append(distribution[i][j])

    # Get True non matching predictions for given threshold
        for i in range(0, distribution.shape[0]):
            for j in range(0, distribution.shape[1]):
                if i != j and distribution[i][j] > x:
                    true_nonmatch.append(distribution[i][j])
    
        false_match.append(len(auth_dist) - len(true_match))
        true_match_dist.append(len(true_match))
        false_nonmatch_dist.append(len(false_nonmatch))
        true_nonmatch_dist.append(len(true_nonmatch))

    far = []
    frr = []

    print("Calculated distributions")
    # Calculate false match rate
    for i,j in zip(false_nonmatch_dist,true_match_dist):
        frr.append((i/(i+j)))
    
#Calculate False acceptance Rate
    for i,j in zip(false_match,true_nonmatch_dist):
        far.append((i/(i+j)))
    
    

    plt.plot(far, frr,'-o')
    plt.xlabel("False Acceptance Rate")
    plt.ylabel("False Rejection Rate")
    plt.title("ROC Curve")
    plt.show()
    
    #CMC curves
    k = 10
    probe_scores = np.zeros((distribution.shape[0],k))
    threshold = 0.494
    
    for i in range(distribution.shape[0]):
        temp =  distribution[i]
        temp = sorted(temp)
        for j in range(0,k):
            probe_scores[i][j] = temp[j]
            
    
    probe_scores = np.transpose(probe_scores)
    
    true_positive_rates = []
    ranks = list(range(1, 11))
    
    for j in range(probe_scores.shape[0]):
        positvie_rates = []
        temp = probe_scores[j]
        for i in temp:
            if i < threshold:
                positvie_rates.append(i)
        
        true_positive_rates.append(len(positvie_rates)/len(temp))
        
        
    
    plt.plot(ranks, true_positive_rates)
    plt.xlabel('Ranks')
    plt.ylabel('True_positive_rates')
    plt.show()
    
    
    
    