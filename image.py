#How do you convert a color image to grayscale? 
#If each color pixel is described by a triple (R, G, B) of intensities for red, green, and blue, 
#how do you map that to a single number giving a grayscale value?

#3 methods
#The lightness method averages the most prominent and least prominent colors: (max(R, G, B) + min(R, G, B)) / 2.

#The average method simply averages the values: (R + G + B) / 3.

#The luminosity method is a more sophisticated version of the average method.
# It also averages the values, but it forms a weighted average to account for human perception. 
#We’re more sensitive to green than other colors, so green is weighted most heavily. 
#The formula for luminosity is 0.21 R + 0.72 G + 0.07 B.

#GRAYSCALE IS USUALLY REFERRED TO INTENSITY
# in grayscale, the pixel values range purely from lowest (darkest) to highest (brightest).

#WHY DO WE CONVERT BRG TO GRAY???????
#https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale

import numpy as np
import cv2 as cv
#Connected components labeling scans an image and groups its pixels into components based on pixel connectivity, 
#i.e. all pixels in a connected component share similar pixel intensity values and are in some way connected with each other.
def get_cc_im(inp_im, disk_sz = [5, 6, 7, 8, 9]):
    "Returns the number of connected components of a difference image"
    #the number of colors used is equal to number of connected components 
    if len(inp_im.shape) == 3:
        # convert to grayscale
        inp_im = cv.cvtColor(inp_im, cv.COLOR_BGR2GRAY)
    
    # ensure single channel is recieved
    assert len(inp_im.shape) == 2
    ret, thresh = cv.threshold(inp_im, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    toret = list()
    for sz in disk_sz:
        struct_elem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (sz, sz))
        opened_im = cv.morphologyEx(thresh, cv.MORPH_OPEN, struct_elem)
        num_comp, _ = cv.connectedComponents(opened_im, connectivity = 8)
        toret.append(num_comp)

    print (toret)
    return toret



#Image registration is the process of transforming different sets of data into one coordinate system.
#Data may be multiple photographs, data from different sensors, times, depths, or viewpoints.

#Suppose we are given two images taken at different times of the same object. To observe different times of the same object
#To observe the changes between these two images, we  need to make sure that they are aligned properly.To obtain this goal,
#we need to find the correct mapping function between the two. The determination of the mapping functions between determination
#of the mapping functions between two images is known as the registration problem

#call from image_change function in this file only
def image_warp(im2, im1, num_iter = 1000):
    "Image registration"

    #BGR to GRAY conversion
    #im1 and im2 must be scalar or numpy array
    #BGR is 3d and graycode is 2d
    #here we have nothing to do with colors so convert it in gray code
    im1_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2,cv.COLOR_BGR2GRAY)

    sz = im1.shape
    #img.shape. It returns a tuple of number of rows, columns and channels (if image is color):
    #bgr image (1080, 1920, 3)
    #graycode (1080, 1920)

    #numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
    #An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.
    #N : int Number of rows in the output.M : int, optional Number of columns in the output. If None, defaults to N.
    #k : int, optional ::Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    #dtype : data-type, optional Data-type of the returned array.

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    #output of np.eye
    #[[ 1.  0.  0.]
 	#[ 0.  1.  0.]]
    
    termination_eps = 1e-5;
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, num_iter,  termination_eps)
 
    # Run the ECC algorithm. The results are stored in warp_matrix.
    #image alignment
    (cc, warp_matrix) = cv.findTransformECC (im1_gray, im2_gray, warp_matrix, cv.MOTION_TRANSLATION, criteria)
 
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP);
    
    return im2_aligned



#call from extract data
def image_change(im2, im1):
    "Compute change between frames"
    
    # size check
    #whenever this is false it throws an exception
    #image difference is related to second method of motion detection
    assert im2.shape == im1.shape

    # image registration
    im2 = image_warp(im2, im1)

    im2 = np.array(im2, dtype = 'int32')
    im1 = np.array(im1, dtype = 'int32')

    diff_im = np.abs(im2 - im1)
    diff_im = np.array(diff_im, dtype = 'uint8')  # convert back to 8-bit per channel
    return diff_im

def image_bleed(im, size = 5):
    ##TODO
    pass
