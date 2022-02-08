import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

from Simulator.Constants import HFOV, VFOV, IMG_WIDTH, IMG_HEIGHT, TARGET_POINTS

class CameraSim:

	#constants for https://www.amazon.com/Microsoft-H5D-00013-LifeCam-Cinema/dp/B009CPC6QA
	#info about pinhole https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/LectureNotes/CV/PinholeCamera/PinholeCamera.html
	#also: from Georgia Tech's CS 4476 (Computer Vision): https://www.cc.gatech.edu/~hays/compvision/lectures/02.pdf

	def __init__(self, hFOV=HFOV, vFOV=VFOV, xPix=IMG_WIDTH, yPix=IMG_HEIGHT, targetShape=TARGET_POINTS):
		#horizontal and vertical fov of the camera
		self.hFOV = hFOV #in radians
		self.vFOV = vFOV

		#pixels dimensions of the output image from the camera
		self.xPix = xPix #in pixels
		self.yPix = yPix

		self.xFocal = (xPix / 2) / math.tan(hFOV / 2) #the focal length of the camera in pixels; x and y have 
		self.yFocal = (yPix / 2) / math.tan(vFOV / 2) #slightly different values because pixels are not perfectly square

		self.camHeading = 0
		self.camPos = np.array([0.0, 0.0, 0.0])

		self.targetShape = np.array(targetShape)
	
	def setCamPos(self, xPos, yPos, zPos, heading):
		'''Set the pose of the camera in 3D space
		'''
		self.camPos[0] = xPos
		self.camPos[1] = yPos
		self.camPos[2] = zPos
		self.camHeading = heading

	def convertToPixels(self, point):
		'''Given a point in 3D space, figure out where that point lies on the a 2D image
		'''
		#Note: we're in y-up coords
		camToTarget = point - self.camPos

		pinholeMatrix = np.array([[ self.xFocal, 0, 		  self.xPix / 2], \
								  [ 0, 			 self.yFocal, self.yPix / 2], \
								  [ 0,			 0, 		  1]])

		headingRotationMatrix = np.array([[math.cos(self.camHeading), 0, math.sin(self.camHeading), 0], \
								   		  [0, 1, 0, 0], \
								   		  [-math.sin(self.camHeading), 0, math.cos(self.camHeading), 0]])

		projectedPoint = np.array([camToTarget[0], camToTarget[1], camToTarget[2], 1]) #project into a slice of a 4D enviornment so we can do homogenous coordinate math

		cameraMatrix = np.matmul(pinholeMatrix, headingRotationMatrix)

		imgPos = np.matmul(cameraMatrix, projectedPoint) #* (1 / (camToTarget[2]))
		
		if imgPos[2] < 1e-5:
			return None

		return [imgPos[0] / imgPos[2], imgPos[1] / imgPos[2]] #convert back from homogenenous coordinates and return value on the screen
	
	def getFrame(self):
		'''Get a binary image of the target taken by the camera
		'''
		pxTargetPoints = []
		for point in self.targetShape:
			pixelPt = self.convertToPixels(point)
			if pixelPt != None:
				pxTargetPoints.append(pixelPt)

		blankImg = np.zeros((self.yPix, self.xPix))
		if len(pxTargetPoints) != len(self.targetShape):
			return blankImg

		pxTargetPoints = np.array(pxTargetPoints, dtype=np.int32)

		area = 0
		if len(pxTargetPoints) > 0:
			img = cv2.fillPoly(blankImg, pts=[pxTargetPoints], color=255)

		return img


if __name__ == '__main__':

	sim = CameraSim()
	
	initTime = time.time()
	sim.setCamPos(0, 0, 0, 0)

	frame = sim.getFrame()
	cv2.imshow("test", np.uint8(frame))
	cv2.waitKey(0)

	