# Image Processing Practice
# Create a variety of filters for an image

# Import opencv and numpy for image handling
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# Load and show image using opencv
img = cv.imread('skyline.jpg', cv.IMREAD_COLOR)

# Convert BGR (opencv standard) to RGB
# This is just a reordering of the array order

#img = cv.resize(img, (int(0.3*img.shape[1]), int(0.3*img.shape[0])))

cv.imshow("Image", img)
#cv.waitKey()

# Convert to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)



# Verify that opencv reads image as numpy array
print(type(img))
# Check shape
print(img.shape)

# Get grayscale
img_gray = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
# Convert to 8 bit
img_gray = img_gray.astype(np.uint8)

cv.imshow("Grayscale", img_gray)
#cv.waitKey()

# Thresholding
thresh = np.array([[255 if x > 150 else 0 for x in row] for row in img_gray])

thresh = thresh.astype(np.uint8)

#cv.imshow("Threshold", thresh)




def applyFilter(img, filter):
    # To maintain size of image, we need padding
    # Add half the size of the filter rows and columns (3 -> add 1 row, 5 --> add 2 rows)
    # Assumes square filter
    filter_size = filter.shape[0]
    zero_col = np.zeros((math.floor(filter_size / 2), img.shape[1]))
    padded = np.vstack((zero_col, img))
    padded = np.vstack((padded, zero_col))
    zero_row = np.zeros((padded.shape[0], filter_size-1))
    padded = np.hstack((zero_row, padded))
    padded = np.hstack((padded, zero_row))
    #Applying filter
    filtered = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):            
            filtered[i][j] = np.sum(np.multiply(padded[i:i+filter_size,j:j+filter_size], filter))
            
    filtered = filtered.astype(np.uint8)
    return filtered

def applySobel(img):
    # Sobel filter in x and y
    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    sobel = applyFilter(img, sobel_x)
    sobel1 = applyFilter(img, sobel_y)
    # Norm of the x and y
    sobel = np.sqrt(sobel**2 + sobel1**2)
    return sobel.astype(np.uint8)



# Average Filter
average_filter = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
blurred = applyFilter(img_gray, average_filter)

cv.imshow("Blurred", blurred)

# Gaussian Blur
gaussian_blur = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
gaussian_blur = 1/256 * gaussian_blur

gaussed = applyFilter(img_gray, gaussian_blur)

cv.imshow("Gaussian Blur", gaussed)


# Edge Detection
edge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edged = applyFilter(img_gray, edge)

#cv.imshow("Edged", edged)

# Sobel Edge Detection
cv.imshow("Sobel from Blur", applySobel(blurred))

cv.imshow("Sobel from Thresh", applySobel(thresh))

cv.imshow("Sobel from Gray", applySobel(img_gray))



# Image Enhancement
unsharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
#unsharp = np.array([[-1/3,-1/3,-1/3], [-1/3,11/3,-1/3], [-1/3,-1/3,-1/3]])
enhanced = applyFilter(blurred, unsharp)

cv.imshow("Enhanced", enhanced)
cv.waitKey(0)










