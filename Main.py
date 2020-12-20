import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import glob

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QMainWindow
from PyQt5.QtGui import QPixmap,QImage,QIcon
from PyQt5 import QtCore, QtGui, QtWidgets

from PYUI import Ui_Form

class MyMainForm(QMainWindow, Ui_Form):
	def __init__(self,parent = None):
		super(MyMainForm, self).__init__(parent)
		self.setupUi(self)
		self.pushButton1_1.clicked.connect(self.Q1_1)
		self.pushButton1_2.clicked.connect(self.Q1_2)
		self.pushButton2_1.clicked.connect(self.Q2_1)
		self.pushButton2_2.clicked.connect(self.Q2_2)
		self.pushButton2_3.clicked.connect(self.Q2_3)
		self.pushButton2_4.clicked.connect(self.Q2_4)
		self.pushButton3.clicked.connect(self.Q3)
		self.pushButton4.clicked.connect(self.Q4)
		self.comboBox.addItems(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])

	def closeEvent(self, event):
		sys.exit(app.quit())

	def Q1_1(self):
		img1 = cv2.imread('./Datasets/Q1_Image/coin01.jpg')
		cv2.imshow('coin01',img1)
		gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		ret , binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
		blurred = cv2.GaussianBlur(binary,(11,11),0)
		edged = cv2.Canny(blurred,30,200)
		contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cv2.drawContours(img1,contours,-1,(0,0,255),3)
		cv2.imshow('Contours_01',img1)

		img2 = cv2.imread('./Datasets/Q1_Image/coin02.jpg')
		cv2.imshow('coin02',img2)
		gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		ret , binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
		blurred = cv2.GaussianBlur(binary,(11,11),0)
		edged = cv2.Canny(blurred,30,200)
		contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cv2.drawContours(img2,contours,-1,(0,0,255),3)
		cv2.imshow('Contours_02',img2)
		cv2.waitKey(0)
		cv2.destroyWindow('coin01')
		cv2.destroyWindow('Contours_01')
		cv2.destroyWindow('coin02')
		cv2.destroyWindow('Contours_02')

	def Q1_2(self):
		img1 = cv2.imread('./Datasets/Q1_Image/coin01.jpg')
		gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		ret , binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
		blurred = cv2.GaussianBlur(binary,(11,11),0)
		edged = cv2.Canny(blurred,30,200)
		contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		text1 = (str(len(contours)))
		self.label_7.setText('There are '+text1+' coins in coin01.jpg' )

		img2 = cv2.imread('./Datasets/Q1_Image/coin02.jpg')
		gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		ret , binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
		blurred = cv2.GaussianBlur(binary,(11,11),0)
		edged = cv2.Canny(blurred,30,200)
		contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		text2 = (str(len(contours)))
		self.label_8.setText('There are ' + text2 + ' coins in coin02.jpg')

	def Q2_1(self):
		w = 11
		h = 8
		objp = np.zeros((w * h, 3), np.float32)
		objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
		objpoints = []
		imgpoints = []
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001)
		images = glob.glob('./Datasets/Q2_Image/*.bmp')
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret , corners = cv2.findChessboardCorners(gray,(w,h),None)

			if ret == True:
				objpoints.append(objp)
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners)
				cv2.drawChessboardCorners(img,(w,h),corners,ret)
				cv2.imshow('img',img)
				cv2.waitKey(700)

		cv2.destroyAllWindows()

	def Q2_2(self):
		w = 11
		h = 8
		objp = np.zeros((w * h, 3), np.float32)
		objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
		objpoints = []
		imgpoints = []
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001)
		images = glob.glob('./Datasets/Q2_Image/*.bmp')
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret , corners = cv2.findChessboardCorners(gray,(w,h),None)

			if ret == True:
				objpoints.append(objp)
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners)
		
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		print(mtx)
	
	def Q2_3(self):
		w = 11
		h = 8
		objp = np.zeros((w * h, 3), np.float32)
		objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
		objpoints = []
		imgpoints = []
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001)
		images = glob.glob('./Datasets/Q2_Image/*.bmp')
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret , corners = cv2.findChessboardCorners(gray,(w,h),None)

			if ret == True:
				objpoints.append(objp)
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners)
		
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)		
		
		i = int(self.comboBox.currentText()) - 1

		Rota = cv2.Rodrigues(rvecs[i])[0]
		tmp0 = np.append(Rota[0],tvecs[i][0][0])
		tmp1 = np.append(Rota[1],tvecs[i][1][0])
		tmp2 = np.append(Rota[2],tvecs[i][2][0])
		res = np.concatenate((tmp0,tmp1))
		res2 = np.concatenate((res,tmp2))
		print(res2)


	def Q2_4(self):
		w = 11
		h = 8
		objp = np.zeros((w * h, 3), np.float32)
		objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
		objpoints = []
		imgpoints = []
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001)
		images = glob.glob('./Datasets/Q2_Image/*.bmp')
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret , corners = cv2.findChessboardCorners(gray,(w,h),None)

			if ret == True:
				objpoints.append(objp)
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners)
		
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		print(dist)
	
	def Q3(self):
		w = 11
		h = 8
		objp = np.zeros((w * h, 3), np.float32)
		objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
		objpoints = []
		imgpoints = []
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001)
		images = glob.glob('./Datasets/Q3_Image/*.bmp')
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret , corners = cv2.findChessboardCorners(gray,(w,h),None)

			if ret == True:
				objpoints.append(objp)
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners)
		
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	
		#draw 
		axis = np.float32([[1,1,0],[3,5,0],[5,1,0],[3,3,-3]])

		def draw_pyramid(img,corners,imgpts):
			imgpts = np.int32(imgpts).reshape(-1,2)
			img = cv2.drawContours(img, [imgpts[:3]],-1,(0,0,255),3)

			img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]),(0,0,255),3)
			img = cv2.line(img, tuple(imgpts[1]), tuple(imgpts[3]),(0,0,255),3)
			img = cv2.line(img, tuple(imgpts[2]), tuple(imgpts[3]),(0,0,255),3)

			img = cv2.drawContours(img, [imgpts[3:]],-1,(0,0,255),3)
			return img
		
		for i in images:
			im = cv2.imread(i)
			gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray,(w,h),None)
			if ret == True:
				_,rvec,tvec,_ = cv2.solvePnPRansac(objp,corners,mtx,dist)
				imgpts, _ = cv2.projectPoints(axis,rvec,tvec,mtx,dist)
				img = draw_pyramid(im,corners,imgpts)
				cv2.imshow('img',img)
				cv2.waitKey(500)
		cv2.destroyAllWindows()
				
	def Q4(self):
		imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
		imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)
		stereo = cv2.StereoBM_create(numDisparities = 512, blockSize = 23)
		disparity = stereo.compute(imgL,imgR)
		disparity = cv2.normalize(disparity,disparity,alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX,dtype = cv2.CV_8U)

		cv2.imshow('result',disparity)

		#select pixal
		img = disparity
		def on_EVENT_LBUTTONDOWN(event,x,y,flags,param):
			if event == cv2.EVENT_LBUTTONDOWN:
				xy = "%d%d" % (x,y)
				cv2.rectangle(img,(2500,1800),(2820,1920),(255,255,255),-1)
				Z = int(178 * 2826 / (disparity[y][x] + 123));
				text1 = 'Disparity: ' + str(int(disparity[y][x])) + ' pixels'
				text2 = 'Depth: ' + str(Z) + ' mm'
				cv2.putText(img,text1,(2500,1830),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
				cv2.putText(img,text2,(2500,1900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
				cv2.imshow('image',img)
		

		cv2.namedWindow('image')
		cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyWindow('result')
		cv2.destroyWindow('image')


		

	
				

		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	myWin = MyMainForm()
	myWin.show()
	sys.exit(app.exec_())

