"""
Muhammad Ahsan Baig's Code
"""









import cv2
import numpy as np
import sys

#Method to remove border of the image
def removeBorder(img):
    #Arbitrarily set border value to 30
    return img[30:len(img)-20 , 30 : len(img[0]) - 20]

#Merges three images from different color channel into one and fix the brightness
#because brightness will increase when pyramids are added up
def getColor(blue,green,red,level):
    #Create a blank image
    img2 = np.zeros((len(red) , len(red[0]) , 3), np.uint8)
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            img2[i][j][0] = blue[i][j]/(level + 1)
            img2[i][j][1] = green[i][j]/(level + 1)
            img2[i][j][2] = red[i][j]/(level + 1)
    return img2

#Returns 3 images from the input image
def getAllFilters(img):
    filters = []
    for k in range(3):
        #Create a blank image of one third height of the original image but with same width
        img2 = np.zeros((len(img) / 3, len(img[0]) - 20, 1), np.uint8)
        for i in range(len(img2)):
            for j in range(len(img2[0])):
                #Lower part of the image is the blue filter
                if k == 0:
                    img2[i][j] = img[i + (2 * len(img2))][j]
                #Middle part of the image is the green filter
                if k == 1:
                    img2[i][j] = img[i + len(img2)][j]
                #Upper portion of the image is red filter
                if k == 2:
                    img2[i][j] = img[i][j]
        filters.append(img2)
    return filters

#Returns a list of images that belongs to each level of the pyramid
def getAllPyramids(img , level):
    G = img.copy()
    gpA = [G]
    for i in xrange(level):
        #Get image in the lower level of the pyramid
        G = cv2.pyrDown(G)
        gpA.append(G)
    return gpA

#Add all the images n the pyramid from smaller to bigger image
def reconstruct(colorPyramid,level):
    #Get top most image i.e smallest
    im = colorPyramid[level]
    for i in range(len(colorPyramid) - 2, -1, -1):
        #Scale up the previous level image
        im = cv2.pyrUp(im)
        #Resize because because pyrUp can give slightly bigger or smaller image than the original pyramid
        #as data is lost in scaling down
        im = cv2.resize(im, (colorPyramid[i].shape[1], colorPyramid[i].shape[0]))
        #Add the Scalled up image and the image on the same level in the original image pyramid
        im = cv2.add(im, colorPyramid[i])
    return im


def task_1(img):
    img2 = np.zeros((len(img) / 3, len(img[0]) - 20, 3), np.uint8)
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            img2[i][j][2] = img[i + (2 * len(img2))][j]
            img2[i][j][1] = img[i + len(img2)][j]
            img2[i][j][0] = img[i][j]
    return img2

level = 4
if len(sys.argv) < 2:
    print "Not Enough Argument"
    sys.exit(0)

img = cv2.imread(sys.argv[1],0)

#########TASK 3 starts########
filters = getAllFilters(img)
pyramids = []
for filter in filters:
    pyramids.append(getAllPyramids(filter,level))
colorPyramid = []
for i in range(len(pyramids[0])):
    colorPyramid.append(getColor(pyramids[2][i],pyramids[1][i],pyramids[0][i],level))

im = reconstruct(colorPyramid,level)
#########TASK 3 ends##########

#########TASK 2 starts########
im = removeBorder(im)
#########TASK 2 ends##########

#Show the final images
cv2.imshow('Image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()



