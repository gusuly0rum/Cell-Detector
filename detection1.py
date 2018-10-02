## Import Necessary Libraries
import os
import cv2
import numpy
import scipy
import matplotlib

import scipy.misc
import scipy.ndimage
import matplotlib.pyplot
import matplotlib.patches
import skimage.morphology

## Function for Binary Filling
def fillHoles(input_image):
    output_image = input_image.copy()
    output_image = output_image.astype(numpy.uint8)
    output_image,contourArray,hierarchy = cv2.findContours(output_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contourArray:
        cv2.drawContours(output_image,[contour],0,255,-1)
    return output_image

## Function for Image Saving
def shapeConvert(input_image,desired_size,minRow,minCol,maxRow,maxCol,minRowNorm=0,minColNorm=0):

    # current sizes
    sizeXCurrent = (minRow + changer) + ((maxRow + changer) - (minRow + changer)) - (minRow + changer)
    sizeYCurrent = (minCol + changer) + ((maxCol + changer) - (minCol + changer)) - (minCol + changer)
    sizeXChange = (desired_size - sizeXCurrent)/2.0
    sizeYChange = (desired_size - sizeYCurrent)/2.0

    # new sizes
    sizeXFrom = (minRow + minRowNorm + changer) - int(numpy.ceil(sizeXChange))
    sizeXTrom = (minRow + minRowNorm + changer) + ((maxRow + changer) - (minRow + changer)) + int(numpy.floor(sizeXChange))
    sizeYFrom = (minCol + minColNorm + changer) - int(numpy.ceil(sizeYChange))
    sizeYTrom = (minCol + minColNorm + changer) + ((maxCol + changer) - (minCol + changer)) + int(numpy.floor(sizeYChange))

    # new image
    outputImage = input_image.copy()
    outputImage = outputImage[sizeXFrom:sizeXTrom, sizeYFrom:sizeYTrom]
    return outputImage


#       [0 1 2  3  4  5  6  7  8  9  10  11  12  13  14  15]
nFile = [6,9,15,30,60,65,68,71,98,99,109,112,113,116,123,135]

## Loop over Each File
for fileIndex in range(0,1):

    ## Set Filenames
    foldername = "C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Data\WBC" + str(nFile[fileIndex])
    filename = "WBC" + str(nFile[fileIndex]) + '_fullimg_1000X_1.jpg'
    os.chdir(foldername)

    ## Color Space Conversion
    # BGR to RGB
    imageBGR = cv2.imread(filename).astype(numpy.uint8)
    imageRGB = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)

    # Crop RGB
    changer = 70
    imagesizeX,imagesizeY,imagesizeZ = imageRGB.shape
    redborder = [1 + changer , 1 + changer , imagesizeX - 2*changer ,  imagesizeY - 2*changer]
    imageRGBCrop = imageRGB[redborder[1]:redborder[1] + redborder[2] + 1 , redborder[0]:redborder[0] + redborder[3] + 1]

    # Cropped RGB to HSV
    imageHSV = cv2.cvtColor(imageRGBCrop,cv2.COLOR_RGB2HSV)
    imageHSV = imageHSV[:,:,2]

    # Cropped RGB to Lab (CIE)
    imageLAB = cv2.cvtColor(imageRGBCrop,cv2.COLOR_RGB2Lab)
    imageLAB = imageLAB[:,:,2]

    # Cropped HSV/Lab to Binary
    thresholdValue,imageBinary = cv2.threshold(imageLAB,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imageLabel = skimage.measure.label(imageBinary)
    propsArray = skimage.measure.regionprops(imageLabel)


    ## Set Axes Properties
    # Create Figure/Axes Instance
    figure,axes = matplotlib.pyplot.subplots(figsize=(10,6))
    matplotlib.pyplot.tight_layout()
    axes.imshow(imageRGB)

    # Set Axes Position
    axesPosition = axes.get_position()
    axesPosition = [axesPosition.x0 , axesPosition.y0 - 0.03, axesPosition.width , axesPosition.height]
    axes.set_position(axesPosition)

    # Set Axes Title
    axes.set_axis_off()
    axes.set_title('WBC' + str(nFile[fileIndex]))

    # Set Red Border
    axes.add_patch(matplotlib.patches.Rectangle((1 + changer , 1 + changer) , imagesizeY - 2*changer ,  imagesizeX - 2*changer,fill=False,edgecolor='red'))

    ## Detection

    numCells = 0

    ## Loop Over Each Object
    for index in range(len(propsArray)):

        # Crop Individual Objects
        imageIndivRGB = imageRGBCrop.copy()
        imageIndivBinary = imageBinary.copy()
        imageIndivBinary[imageLabel != index + 1] = 0
        minRow,minCol,maxRow,maxCol = propsArray[index].bbox
        imageIndivRGB = imageIndivRGB[minRow:minRow + (maxRow - minRow) , minCol:minCol + (maxCol - minCol)]
        imageIndivBinary = imageIndivBinary[minRow:minRow + (maxRow - minRow) , minCol:minCol + (maxCol - minCol)]

        ## Ignore Small Objects and Keep Normal Objects
        if propsArray[index].area < 5000 and propsArray[index].area > 1500:

            # Count Normal Cells
            numCells += 1

            # Binary Operations
            imageFillHole = fillHoles(imageIndivBinary)
            squareKernel = numpy.ones([3,3],numpy.uint8)
            imageOpen = cv2.morphologyEx(imageFillHole , cv2.MORPH_OPEN , squareKernel , iterations=5)
            imageDilate = cv2.dilate(imageOpen , squareKernel , iterations=1)

            # Plot Centroids
            propsArrayNot = skimage.measure.regionprops(imageDilate)
            xPos,yPos = propsArrayNot[0].centroid
            axes.plot(yPos + minCol + changer , xPos + minRow + changer , 'yo')
            axes.text(yPos + minCol + changer , xPos + minRow + changer , str(numCells))

            # Resize Individual Cells for Saving
            imageSave = shapeConvert(imageRGB,148,minRow,minCol,maxRow,maxCol)
            # scipy.misc.imsave(str(numCells) + '.jpg',imageSave)


        ## Split Large Objects
        elif propsArray[index].area > 5000:

            # Obtain Boundaries
            imageBinaryBoundaries = imageIndivBinary - cv2.erode(imageIndivBinary , None , iterations=1)

            # Distance Transform
            imageDistancemap = cv2.distanceTransform(imageIndivBinary,cv2.DIST_L2,5)

            # Obtain Markers
            thresholdValue, imageBinaryMarkers = cv2.threshold(imageDistancemap , 0.5*imageDistancemap.max() , 255 , 0)
            imageLabelMarkers = skimage.measure.label(imageBinaryMarkers)
            imageLabelMarkers = imageLabelMarkers + 1
            imageLabelMarkers[imageBinaryBoundaries == 255] = 0

            # Marker-Based Watershed Transform
            imageWatershed = cv2.watershed(imageIndivRGB,numpy.int32(imageLabelMarkers))
            imageWatershed[imageWatershed == 1] = 0

            # Plot Centroids
            propsArrayWater = skimage.measure.regionprops(imageWatershed)
            for subIndex in range(0,len(propsArrayWater)):
                if propsArrayWater[subIndex].area > 100:
                    numCells += 1
                    xPos,yPos = propsArrayWater[subIndex].centroid
                    axes.plot(yPos + minCol + changer,xPos + minRow + changer,'yo')
                    axes.text(yPos + minCol + changer,xPos + minRow + changer,str(numCells))

                    # Resize Individual Cells for Saving
                    minRowWater,minColWater,maxRowWater,maxColWater = propsArrayWater[subIndex].bbox
                    imageSave = shapeConvert(imageRGB,148,minRowWater,minColWater,maxRowWater,maxColWater,minRow,minCol)
                    # scipy.misc.imsave(str(numCells) + '.jpg',imageSave)

                    # cv2.imshow('Image',imageSave)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()


    matplotlib.pyplot.show()
print 'Number of Cells Detected: ' + str(numCells) + " (WBC" + str(nFile[fileIndex]) + ")"


def drawBoundingBox(input_image,input_array,setIndex=False):

    window_size = [500,800]
    scaleX = window_size[0] / imageRGB.shape[1]
    scaleY = window_size[1] / imageRGB.shape[0]
    windowX = int(imageRGB.shape[1] * min(scaleX,scaleY))
    windowY = int(imageRGB.shape[0] * min(scaleX,scaleY))
    cv2.resizeWindow('Image',windowX,windowY)
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)

    if len(input_array) == 1:
        for index in range(input_array[0],input_array[0] + 1):
            minRow,minCol,maxRow,maxCol = propsArray[index].bbox
            cv2.rectangle(input_image , (minCol,minRow) , (minCol + (maxCol - minCol),minRow + (maxRow - minRow)) , [0,255,0] , 2)
            if setIndex == True:
                cv2.putText(imageRGBCrop , str(index) , (minCol,minRow) , cv2.FONT_HERSHEY_SIMPLEX , fontScale=1 , color=[255,0,0] , thickness=5)
            cv2.imshow('Image',input_image)
    else:
        for index in range(input_array[0],input_array[1]):
            minRow,minCol,maxRow,maxCol = propsArray[index].bbox
            cv2.rectangle(input_image , (minCol,minRow) , (minCol + (maxCol - minCol),minRow + (maxRow - minRow)) , [0,255,0] , 2)
            if setIndex == True:
                cv2.putText(imageRGBCrop , str(index) , (minCol,minRow) , cv2.FONT_HERSHEY_SIMPLEX , fontScale=1 , color=[255,0,0] , thickness=5)
            cv2.imshow('Image',input_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
