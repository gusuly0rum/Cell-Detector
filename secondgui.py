# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy
import scipy
import skimage
import scipy.misc
import skimage.morphology
from PyQt4 import QtGui,QtCore
from firstgui import Ui_MainWindow
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# cd C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Python Scripts
# C:\Python27\Lib\site-packages\PyQt4\pyuic4 firstgui.ui >> firstgui.py

class Login(QtGui.QDialog):
    def __init__(self):

        # create login window
        super(Login,self).__init__()
        self.loginWindow = QtGui.QVBoxLayout(self)

        # set login window properties
        self.setWindowTitle('  Login')
        self.setGeometry(800,300,270,170)
        self.setWindowIcon(QtGui.QIcon('C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Python Scripts\snulogo.png'))

        # create widgets
        self.texteditUsername = QtGui.QLineEdit(self)
        self.texteditPassword = QtGui.QLineEdit(self)
        self.pushbuttonLogin = QtGui.QPushButton('Login',self)
        self.pushbuttonRegister = QtGui.QPushButton('Register',self)

        # set widgets on login window
        self.loginWindow.addWidget(self.texteditUsername)
        self.loginWindow.addWidget(self.texteditPassword)
        self.loginWindow.addWidget(self.pushbuttonLogin)
        self.loginWindow.addWidget(self.pushbuttonRegister)

        # set default user credentials
        self.texteditUsername.setText('melab321')
        self.texteditPassword.setText('melabmelab')

        # specify push button behavior
        self.pushbuttonLogin.clicked.connect(self.checkCredentials)
        self.pushbuttonRegister.clicked.connect(self.createAccount)


    def checkCredentials(self):
        if self.texteditUsername.text() == 'melab321' and self.texteditPassword.text() == 'melabmelab':
            self.accept()

    def createAccount(self):
        pass



class Main(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.home()

    def home(self):
        self.displayMessages(1,0,2,3,0,7)
        self.setWindowIcon(QtGui.QIcon('C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Python Scripts\snulogo.png'))

        # set menubar triggers
        self.ui.menubar_about.triggered.connect(self.menubarAbout)
        self.ui.menubar_openFile.triggered.connect(self.menubarOpenFile)
        self.ui.menubar_runModule.triggered.connect(self.menubarRunModule)
        self.ui.menubar_runClassification.triggered.connect(self.menubarRunClassification)
        self.ui.menubar_restartApplication.triggered.connect(self.menubarRestartApplication)
        self.ui.menubar_saveImage.triggered.connect(self.menubarSaveImage)

        # set push button triggers
        self.ui.pushButton_loadImage.clicked.connect(self.menubarOpenFile)
        self.ui.pushButton_runDetection.clicked.connect(self.menubarRunModule)
        self.ui.pushButton_runClassification.clicked.connect(self.menubarRunClassification)

        # set comboBox triggers
        self.ui.comboBox_dataOptions.activated.connect(self.setPlotWidgetData)
        self.ui.comboBox_dataOptions.activated.connect(self.setTreeWidgetData)

        # set image display triggers
        # self.ui.label_imageDisplay.mousePressEvent = self.drawBoundingBox

        # set display bounding boxes triggers
        self.ui.checkBox_displayIndices.stateChanged.connect(self.displayIndices)

        # set display bounding boxes triggers
        self.ui.checkBox_displayBoundingBoxes.stateChanged.connect(self.displayBoundingBoxes)

        # set cell display trigger
        self.ui.spinBox_cellDisplay.valueChanged.connect(self.cellDisplay)

        # add matplotlib widget
        self.setMenuWidgetProperties()
        self.setPlotWidgetProperties()
        self.setTreeWidgetProperties()

    def setMenuWidgetProperties(self):
        self.ui.comboBox_dataOptions.addItem("Summary")
        for index in range(1,11):
            comboBoxOption = "Class" + str(index)
            self.ui.comboBox_dataOptions.addItem(comboBoxOption)

    def setPlotWidgetProperties(self):
        # create axes instance
        self.figureInstance = Figure()
        self.canvasInstance = FigureCanvas(self.figureInstance)
        self.ui.verticalLayout.addWidget(self.canvasInstance)
        self.axesInstance = self.figureInstance.add_subplot(111)

        # set general histogram properties
        self.axesInstance.set_xlim([0,11])
        self.axesInstance.set_xticks(scipy.arange(1,11,1))
        self.axesInstance.spines['top'].set_visible(False)
        self.axesInstance.spines['left'].set_visible(False)
        self.axesInstance.spines['right'].set_visible(False)
        # self.axesInstance.spines['bottom'].set_visible(False)
        self.axesInstance.tick_params(axis='both',which='both',top='off',bottom='off',labelbottom='on',left='off',right='off',labelleft='off')

        # self.setPlotWidgetData()

    def setTreeWidgetProperties(self):
        self.ui.treeWidget_cellData.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        # self.setTreeWidgetData()

    def setPlotWidgetData(self):
        self.axesInstance.cla()

        # obtain histogram data
        self.classArray = [1,2,3,4,5,6,7,8,9,10]
        self.summaryData = [10,15,5,0,20,30,17,2,3,21]
        self.softmaxData = []
        self.softmaxData.append([0,0,0,50,100,0,0,0,0,0])
        self.softmaxData.append([0,0,70,0,0,0,90,0,0,0])
        self.softmaxData.append([0,0,100,0,0,0,0,0,0,0])
        self.softmaxData.append([0,0,100,0,0,0,0,0,0,0])
        self.softmaxData.append([0,0,0,100,0,0,0,0,0,0])
        self.softmaxData.append([0,0,0,0,0,100,0,0,0,0])
        self.softmaxData.append([0,0,0,0,0,0,0,100,0,0])
        self.softmaxData.append([0,0,0,0,0,0,0,0,100,0])
        self.softmaxData.append([0,0,0,0,0,0,0,0,0,100])
        self.softmaxData.append([0,0,0,0,0,0,0,0,0,100])
        self.softmaxData.append([0,0,0,0,0,0,0,0,0,100])

        # set option-dependent histogram properties
        if self.ui.comboBox_dataOptions.currentText() == self.ui.comboBox_dataOptions.itemText(0):
            self.axesInstance.set_ylim([0,numpy.max(self.summaryData) + 3])
            self.ui.label_xAxisTitle.setText('Frequency [n]   vs   Class Label [k]')
            self.ui.label_xAxisTitle.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.setPlotData(self.summaryData)
        else:
            self.axesInstance.set_ylim([0,100 + 3])
            self.ui.label_xAxisTitle.setText('Softmax [%]   vs   Class Label [k]')
            self.ui.label_xAxisTitle.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            menuIndex = self.ui.comboBox_dataOptions.currentIndex()
            self.setPlotData(self.softmaxData[menuIndex])

    def setPlotData(self,histogramData):
        # set histogram data
        self.axesInstance.set_xlim([0,11])
        self.axesInstance.set_xticks(scipy.arange(1,11,1))
        self.axesInstance.bar(self.classArray,histogramData,align='center')
        binPatches = self.axesInstance.patches
        for binPatch,frequencyValue in zip(binPatches,histogramData):
            height = binPatch.get_height()
            self.axesInstance.text(binPatch.get_x() + binPatch.get_width()/2 , height + 1 , frequencyValue , ha='center' , va='bottom' , fontSize=10)
        self.canvasInstance.draw()

    def setTreeWidgetData(self):

        self.ui.treeWidget_cellData.clear()
        for element in self.classArray:
            QtGui.QTreeWidgetItem(self.ui.treeWidget_cellData).setText(0,str(element))

        # set option-dependent tree properties
        if self.ui.comboBox_dataOptions.currentText() == self.ui.comboBox_dataOptions.itemText(0):
            headerArray = ['Class [k]','Cell Count [n]','Proportion [%]']
            self.ui.treeWidget_cellData.setHeaderLabels(headerArray)
            self.ui.treeWidget_cellData.setColumnCount(len(headerArray))
            self.ui.treeWidget_cellData.setColumnWidth(0,160)
            self.ui.treeWidget_cellData.setColumnWidth(1,160)
            self.ui.treeWidget_cellData.setColumnWidth(2,150)
            self.setTreeData(self.summaryData)
        else:
            headerArray = ['Class [k]','Softmax Probability [%]']
            self.ui.treeWidget_cellData.setHeaderLabels(headerArray)
            self.ui.treeWidget_cellData.setColumnCount(len(headerArray))
            self.ui.treeWidget_cellData.setColumnWidth(0,200)
            self.ui.treeWidget_cellData.setColumnWidth(1,200)
            menuIndex = self.ui.comboBox_dataOptions.currentIndex()
            self.setTreeData(self.softmaxData[menuIndex])


    def setTreeData(self,histogramData):
        self.ui.treeWidget_cellData.clear()

        # set option-dependent tree properties
        if self.ui.comboBox_dataOptions.currentText() == self.ui.comboBox_dataOptions.itemText(0):
            totalCellCount = float(numpy.sum(histogramData))
            for index,element in enumerate(histogramData):
                treeItemInstance = QtGui.QTreeWidgetItem(self.ui.treeWidget_cellData)
                treeItemInstance.setText(0,str(index + 1))
                treeItemInstance.setText(1,str(element))
                treeItemInstance.setText(2,str(numpy.round(element/totalCellCount*100,1)))
        else:
            for index,element in enumerate(histogramData):
                treeItemInstance = QtGui.QTreeWidgetItem(self.ui.treeWidget_cellData)
                treeItemInstance.setText(0,str(index + 1))
                treeItemInstance.setText(1,str(element))



    def menubarAbout(self):
        import webbrowser
        webbrowser.open("http://melab.snu.ac.kr/melab/doku.php")

    def menubarOpenFile(self):
        from PyQt4.QtGui import QFileDialog
        os.chdir("C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Data\WBC135")
        self.filename = QFileDialog.getOpenFileName(self,'Open File')
        if self.filename:

            # convert file to pixmap
            self.displayMessages(0,6)
            self.pixmap_image = QtGui.QPixmap(self.filename)
            self.imageWidth = self.pixmap_image.size().width()
            self.imageHeight = self.pixmap_image.size().height()
            changer = 70

            # set pen properties
            self.penRedBorder = QtGui.QPen(QtCore.Qt.red)
            self.penRedBorder.setWidth(3)

            self.penCentroid = QtGui.QPen(QtCore.Qt.yellow)
            self.penCentroid.setCapStyle(QtCore.Qt.RoundCap)
            self.penCentroid.setWidth(15)

            self.penBoundingBox = QtGui.QPen(QtCore.Qt.yellow)
            self.penBoundingBox.setWidth(2)

            self.penIndex = QtGui.QPen(QtCore.Qt.yellow)
            self.penIndex.setWidth(2)

            # set red border
            self.painterInstance = QtGui.QPainter(self.pixmap_image)
            self.painterInstance.setPen(self.penRedBorder)
            self.painterInstance.drawRect(changer , changer , self.imageWidth - 2*changer ,  self.imageHeight - 2*changer)

            # set label properties
            self.ui.label_imageDisplay.setPixmap(self.pixmap_image)
            self.ui.label_imageDisplay.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.label_imageDisplay.setScaledContents(True)
            self.ui.label_imageDisplay.setMinimumSize(1,1)
            self.ui.label_imageDisplay.show()

            # set current path label
            self.ui.textBrowser_currentPath.setText(self.filename)

            self.imagesaveDirectory()

        else:
            return

    def menubarSaveImage(self):
        from PyQt4.QtGui import QFileDialog
        os.chdir("C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Data\WBC135")
        # self.filename = QFileDialog.getSaveFolderName(self,'Save File')
        self.foldername = str(QFileDialog.getExistingDirectory(self,'Image Save Path')) + "\\"
        if self.foldername:
            self.ui.textBrowser_imagePath.setText(self.foldername)
            self.displayMessages(0,9)
        else:
            return

    def imagesaveDirectory(self):
        self.foldername = os.path.abspath(os.path.join(str(self.filename),os.pardir))
        os.chdir(self.foldername)
        if not os.path.exists(str(self.foldername) + '\subimages'):
            self.foldername += '\subimages\\'
            os.makedirs(self.foldername)
        else:
            self.foldername += '\subimages\\'
        os.chdir(self.foldername)
        self.ui.textBrowser_imagePath.setText(self.foldername)
        self.displayMessages(9)


    def menubarRestartApplication(self):
        pass

    def displayMessages(self,*index):
        status_messages = ["",
                           "Welcome to CellPy!",
                           "Seoul National University,",
                           "Medical Electronics Laboratory",
                           ">> Attribute Error: Image file was not assigned to instance variable",
                           ">> Running cell detection...",
                           ">> Image uploaded",
                           "-"*170,
                           "Cell detection complete",
                           "Image save directory specified",
                           ">> Attribute Error: Image save directory was not assigned to instance variable",
                           "Cell images saved"]
        for index,element in enumerate(index):
            self.ui.textBrowser_status.append(status_messages[element])

    def menubarRunModule(self):
        if not hasattr(self,'filename'):
            self.displayMessages(0,4)
        elif not hasattr(self,'foldername'):
            self.displayMessages(0,10)
        else:
            self.displayMessages(0,5)
            self.detectCells()

    def detectCells(self):

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

        ## Color Space Conversion
        # BGR to RGB
        imageBGR = cv2.imread(str(self.filename)).astype(numpy.uint8)
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

        self.indicesArray = []
        self.boundingBoxArray = []
        self.imageSaveArray = []

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

                self.painterInstance.setPen(self.penCentroid)
                self.painterInstance.drawPoint(yPos + minCol + changer , xPos + minRow + changer)

                self.painterInstance.setPen(self.penIndex)
                self.painterInstance.setFont(QtGui.QFont('Decorative',20))
                # self.painterInstance.drawText(yPos + minCol + changer , xPos + minRow + changer , str(numCells))
                self.ui.label_imageDisplay.setPixmap(self.pixmap_image)

                # Resize Individual Cells for Saving
                imageSave = shapeConvert(imageRGB,148,minRow,minCol,maxRow,maxCol)
                self.imageSaveArray[numCells] = imageSave
                scipy.misc.imsave(self.foldername + str(numCells) + '.jpg',imageSave)

                # append bounding box dimensions to instance variable array
                self.boundingBoxArray.append([minRow,minCol,maxRow,maxCol])
                self.indicesArray.append([yPos + minCol + changer , xPos + minRow + changer])


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

                        self.painterInstance.setPen(self.penCentroid)
                        self.painterInstance.drawPoint(yPos + minCol + changer , xPos + minRow + changer)

                        self.painterInstance.setPen(self.penIndex)
                        self.painterInstance.setFont(QtGui.QFont('Decorative',20))
                        # self.painterInstance.drawText(yPos + minCol + changer , xPos + minRow + changer , str(numCells))
                        self.ui.label_imageDisplay.setPixmap(self.pixmap_image)

                        # Resize Individual Cells for Saving
                        minRowWater,minColWater,maxRowWater,maxColWater = propsArrayWater[subIndex].bbox
                        imageSave = shapeConvert(imageRGB,148,minRowWater,minColWater,maxRowWater,maxColWater,minRow,minCol)
                        self.imageSaveArray[numCells] = imageSave
                        scipy.misc.imsave(self.foldername + str(numCells) + '.jpg',imageSave)

                        # append bounding box dimensions to instance variable array
                        self.boundingBoxArray.append([minRowWater + minRow , minColWater + minCol , maxRowWater + minRow , maxColWater + minCol])
                        self.indicesArray.append([yPos + minCol + changer , xPos + minRow + changer])

        # Display message
        self.displayMessages(8,11)
        self.setPlotWidgetData()
        self.setTreeWidgetData()

    def displayIndices(self):
        if self.ui.checkBox_displayIndices.isChecked():
            for index in range(len(self.indicesArray)):
                indicesList = self.indicesArray[index]
                minRow = indicesList[1]
                minCol = indicesList[0]
                self.painterInstance.setPen(self.penIndex)
                self.painterInstance.drawText(minCol + 5 , minRow - 5 , str(index))
                self.ui.label_imageDisplay.setPixmap(self.pixmap_image)
        else:
            pass

    def displayBoundingBoxes(self):
        if self.ui.checkBox_displayBoundingBoxes.isChecked():
            changer = 70
            for index in range(len(self.boundingBoxArray)):
                boundingBoxList = self.boundingBoxArray[index]
                minRow = boundingBoxList[0]
                minCol = boundingBoxList[1]
                maxRow = boundingBoxList[2]
                maxCol = boundingBoxList[3]
                self.painterInstance.setPen(self.penBoundingBox)
                self.painterInstance.drawRect(minCol + changer , minRow + changer , maxRow - minRow , maxCol - minCol)
                self.ui.label_imageDisplay.setPixmap(self.pixmap_image)
        else:
            pass
        # self.painterInstance.end()

    def cellDisplay(self):
        cellImage = self.imageSaveArray[0]
        self.pixmap_imageSave = QtGui.QImage(cellImage,148,148,QtGui.QImage.Format_RGB32)
        self.painterInstance = QtGui.QPainter(self.pixmap_imageSave)

        # set label properties
        self.ui.label_cellDisplay.setPixmap()
        self.ui.label_cellDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_cellDisplay.setScaledContents(True)
        self.ui.label_cellDisplay.setMinimumSize(1,1)
        self.ui.label_cellDisplay.show()


    # def mousePressEvent(self,QMouseEvent):
    #     return QMouseEvent.pos()
    #
    # def mouseReleaseEvent(self,QMouseEvent):
    #     cursor = QtGui.QCursor()
    #     return cursor.pos()
    #
    # def drawBoundingBox(self,event):
    #     print self.ui.mousePressEvent(event)
    #     print self.ui.mouseReleaseEvent(event)


    def menubarRunClassification(self):
        pass



if __name__ == '__main__':
    mainApplication = QtGui.QApplication(sys.argv)
    loginApplication = Login()
    if loginApplication.exec_() == QtGui.QDialog.Accepted:
        windowFigure = Main()
        windowFigure.move(200,30)
        windowFigure.show()
        sys.exit(mainApplication.exec_())

# if __name__ == '__main__':
#     application = QtGui.QApplication(sys.argv)
#     windowFigure = Main()
#     windowFigure.move(200,30)
#     windowFigure.show()
#     sys.exit(application.exec_())


# cd C:\Users\sjdml\Documents\Seoul National University\User Interface Project\Python Scripts
# C:\Python27\Lib\site-packages\PyQt4\pyuic4 firstgui.ui >> firstgui.py
