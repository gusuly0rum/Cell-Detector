import cv2
import numpy
import scipy
import skimage
import pymorph

import scipy.misc
import scipy.ndimage
import skimage.morphology


## Read File
# BGR to RGB
filepath = "C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Data\wbc123\WBC123_fullimg_1000X_1.jpg"
imageBGR = cv2.imread(filepath)
imageBGR = imageBGR.astype(numpy.uint8)
imageRGB = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)


## Color Space Conversion

# Crop RGB
changer = 70
imagesizeX,imagesizeY,imagesizeZ = imageRGB.shape
redborder = [1 + changer , 1 + changer , imagesizeX - 2*changer ,  imagesizeY - 2*changer]
imageRGBCrop = imageRGB[redborder[1]:redborder[1] + redborder[2] + 1 , redborder[0]:redborder[0] + redborder[3] + 1]

# Cropped RGB to HSV
imageHSV = cv2.cvtColor(imageRGBCrop,cv2.COLOR_RGB2HSV)
def normalizeNDMatrix (input_matrix):
    channelA,channelB,channelC = numpy.dsplit(input_matrix,3)
    flatA = numpy.ndarray.flatten(channelA).astype(numpy.double)
    flatB = numpy.ndarray.flatten(channelB).astype(numpy.double)
    flatC = numpy.ndarray.flatten(channelC).astype(numpy.double)
    channelA,channelB,channelC = channelA/max(flatA),channelB/max(flatB),channelC/max(flatC)
    channelB = channelB - 0.0747
    return channelA,channelB,channelC

channelH,channelS,channelV = normalizeNDMatrix(imageHSV)

# Cropped RGB to Lab (CIE)
imageGaussFilt = cv2.GaussianBlur(imageRGBCrop,(3,3),3)
imageLAB = cv2.cvtColor(imageGaussFilt,cv2.COLOR_RGB2Lab)
channelL,channelA,channelB = numpy.dsplit(imageLAB,3)
meanLDim = numpy.squeeze(numpy.asarray(channelL)).mean()
meanADim = numpy.squeeze(numpy.asarray(channelA)).mean()
meanBDim = numpy.squeeze(numpy.asarray(channelB)).mean()
imageLABContrast = (channelL - meanLDim)**2 + (channelA - meanADim)**2 + (channelB - meanBDim)**2
emptyMatrix = numpy.zeros(imageLABContrast.shape)
imageLABGray = cv2.normalize(imageLABContrast,emptyMatrix,1.0,0.0,cv2.NORM_MINMAX)


## Saliency Map Creation
# Resize LAB Image
cropsizeX,cropsizeY,cropsizeZ = imageLABGray.shape
downscaleTuple = (int(numpy.ceil(cropsizeX/6.0)),int(numpy.ceil(cropsizeY/6.0)))
imageLABResize = cv2.resize(imageLABGray,downscaleTuple)]

# Spectral Residual
imageLABFFT = numpy.fft.fft2(imageLABResize)
imageLABPhase = numpy.angle(imageLABFFT)
imageLABLogAmp = numpy.log(numpy.abs(imageLABFFT))
rectAverageKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))/9.0
imageLABAverageFilt = scipy.ndimage.correlate(imageLABLogAmp,rectAverageKernel,mode='constant')
imageLABSpectralResidual = imageLABLogAmp - imageLABAverageFilt
imageLABSaliencyMap = imageLABSpectralResidual + 1j*imageLABPhase
imageLABSaliencyMap = numpy.exp(imageLABSpectralResidual + 1j*imageLABPhase)
imageLABSaliencyMap = numpy.fft.ifft2(imageLABSaliencyMap)
imageLABSaliencyMap = numpy.abs(imageLABSaliencyMap)**2

# After Effect
def createDiskKernel(radius):
    diskShape = 2*radius + 1
    diskAverageKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diskShape,diskShape))
    return diskAverageKernel

imageLABSaliencyMap =cv2.filter2D(imageLABSaliencyMap,-1,createDiskKernel(3))
imageLABSaliencyMap = cv2.normalize(imageLABSaliencyMap,emptyMatrix,1.0,0.0,cv2.NORM_MINMAX)

# Binary Image
def fillHoles(input_image):
    output_image = input_image.copy()
    output_image = output_image.astype(numpy.uint8)
    output_image,contourArray,hierarchy = cv2.findContours(output_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contourArray:
        cv2.drawContours(output_image,[contour],0,255,-1)
    return output_image

imageLABBinary = scipy.misc.imresize(imageLABSaliencyMap,(cropsizeX,cropsizeY))
imageLABBinary = imageLABBinary > imageLABBinary.mean()
imageLABFill = fillHoles(imageLABBinary)


## Cell Properties Extraction
imageLABLabel,numCells = cv2.connectedComponents(imageLABFill)
imageLABLabel = numpy.uint8(imageLABLabel)
imageLABLabel,contourArray,hierarchy = cv2.findContours(imageLABLabel,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


## Cell Detection
def neighbors(input_matrix,input_array):
    output_array = [0]*4
    indexRow = input_array[0]
    indexCol = input_array[1]
    nRows,nCols = input_matrix.shape
    output_array[0] = input_matrix[(indexRow - 1) % nRows,indexCol]
    output_array[1] = input_matrix[indexRow,(indexCol + 1) % nCols]
    output_array[2] = input_matrix[(indexRow + 1) % nRows,indexCol]
    output_array[3] = input_matrix[indexRow,(indexCol - 1) % nCols]
    return output_array

def bwmorph(input_matrix):
    output_matrix = input_matrix.copy()
    if len(output_matrix.shape) == 3:
        output_matrix = output_matrix[:,:,0]
    nRows,nCols = output_matrix.shape
    original_matrix = output_matrix.copy()
    for indexRow in range(0,nRows):
        for indexCol in range(0,nCols):
            center_pixel = [indexRow,indexCol]
            neighbor_array = neighbors(original_matrix,center_pixel)
            if numpy.all(neighbor_array):
                output_matrix[indexRow,indexCol] = 0
    return output_matrix

def hminima(input_matrix,input_integer):
    output_matrix = numpy.max(input_matrix) - input_matrix
    output_matrix = skimage.morphology.reconstruction(output_matrix - input_integer , output_matrix)
    output_matrix = numpy.max(input_matrix) - output_matrix
    return output_matrix

def intensity_based_watershed(mask,intensitymap,h):
    intensitymap = numpy.multiply(intensitymap,mask)
    boundaries = bwmorph(mask)
    distancemap = numpy.multiply(cv2.distanceTransform(boundaries , cv2.DIST_L2 , cv2.DIST_MASK_PRECISE) , -1)
    distancemap = numpy.multiply(distancemap,mask)
    distancemap[distancemap == 0] = 1
    distancemap = hminima(distancemap,h)


    # scale image
    window_size = [500,800]
    scaleX = window_size[0] / imageRGB.shape[1]
    scaleY = window_size[1] / imageRGB.shape[0]
    windowX = int(imageRGB.shape[1] * min(scaleX,scaleY))
    windowY = int(imageRGB.shape[0] * min(scaleX,scaleY))
    # show image
    cv2.resizeWindow('Image',windowX,windowY)
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image',minima)
    cv2.waitKey()
    cv2.destroyAllWindows()

# for index in range(0,len(contourArray)):
#[14,23]
# contourArray is sorted from largest object to smallest object
index = 11

xPos,yPos,xLen,yLen = cv2.boundingRect(contourArray[index])
bboxLAB = [xPos,yPos,xLen,yLen]
imageHSVIndiv = channelS.copy()[:,:,0] * numpy.equal(imageLABLabel,numpy.max(imageLABLabel) - index)
# cv2.rectangle(imageHSVIndiv , (xPos,yPos) , (xPos + xLen , yPos + yLen) , [255,0,0] , 2)
imageHSVIndivCrop = imageHSVIndiv[yPos:yPos + yLen , xPos:xPos + xLen]

imageHSVIndivOpen = cv2.morphologyEx(imageHSVIndivCrop , cv2.MORPH_OPEN , createDiskKernel(10))
imageHSVIndivErode = cv2.erode(imageHSVIndivOpen , createDiskKernel(10) , iterations=1)
imageHSVIndivRecon = skimage.morphology.reconstruction(imageHSVIndivErode,imageHSVIndivCrop)
imageHSVIndivClose = cv2.morphologyEx(imageHSVIndivOpen , cv2.MORPH_CLOSE , createDiskKernel(20))
imageHSVIndivDilate = cv2.dilate(imageHSVIndivRecon , createDiskKernel(20) , iterations=1)
imageHSVIndivComple = 255 - skimage.morphology.reconstruction(255 - imageHSVIndivDilate , 255 - imageHSVIndivRecon)
imageHSVIndivComple[imageHSVIndivComple != 0] = 1
imageHSVIndivBinary = imageHSVIndivComple
imageHSVIndivFill = fillHoles(imageHSVIndivBinary)

imageHSVIndivWater = intensity_based_watershed(imageHSVIndivFill,imageHSVIndivCrop,10)
# Display Bounding Boxes
# figure,axes = matplotlib.pyplot.subplots(figsize=(10,6))
# matplotlib.pyplot.tight_layout()
# axes.imshow(imageRGBCrop)
# axes.set_axis_off()
#
# for index in range(25,26):
#     minRow,minCol,maxRow,maxCol = propsArray[index].bbox
#     boundingBox = matplotlib.patches.Rectangle((minCol,minRow),maxCol - minCol,maxRow - minRow,fill=False,edgeColor='red',linewidth=1)
#     axes.add_patch(boundingBox)
#     print [int(minCol),int(minRow)]
#     print maxCol - minCol
#     print maxRow - minRow
#
# matplotlib.pyplot.show()


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
