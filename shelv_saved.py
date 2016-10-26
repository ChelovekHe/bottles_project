import glob
#from scipy.ndimage.measurements import label
from numpy import where, shape, zeros, dot, array
from skimage.io import imread,imsave
from skimage.morphology import closing, rectangle, disk
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.filters import threshold_adaptive
from skimage.filters.rank import median
from math import trunc
import cv2
import numpy as np
import os

# Функция превращает изображение в полутоновое
def rgb2gray (rgb):

    return dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Бинаризация
def binar (gray_img):

	gray_img[where(gray_img>=0.5)]=1
	gray_img[where(gray_img<0.5)]=0

	return gray_img
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def list_of_shelves (props,rows,cols):
	
	list_of_true = []
	tmp = []

	for i in range(len(props)):
			
			t,y,r,w=props[i].bbox
			if w-y >= 0.8*cols:
				#print (i)
				tmp.append(t)
				tmp.append(r)
				list_of_true.append(tmp)
				tmp = []

	return list_of_true  

def show_me_lab (rows,cols,labelok,label_n):

	sp_lab = zeros((rows,cols), dtype='uint8')
	for i in range (rows):
		for j in range(cols):
			
			if (labelok[i][j]==label_n):
				sp_lab[i][j] = 255
			
			else:
				sp_lab[i][j] = 0

	cv2.imwrite("label.jpg", sp_lab) 

def main():
	key_word='testdata'+'/'
	file_path1='/home/ruslan/'+key_word
	for name in os.listdir(file_path1):
		print(name)
		image = cv2.imread(key_word+name)
		img = imread(key_word+name)
		saved = img.copy()
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		gray_x = cv2.Sobel(blurred, ddepth=-1, dx=1, dy=0)
		gray_y = cv2.Sobel(blurred, ddepth=-1, dx=0, dy=1)
		
		row=gray_x.shape[0]
		col=gray_x.shape[1]

		a=np.zeros((row,col),dtype='int')
		a+=2
		gray = gray_x**a + gray_y**a
		a = a.astype("float32") - 1.5
		gray = np.trunc(gray ** a).astype("int")
		
		middle = gray.sum()/(row*col)
		
		gray[np.where(gray<middle)]=0
		gray[np.where(gray>=middle)]=255

		label1 = label(gray,neighbors = 8)

		#Получение данных для каждого из помеченных объектов
		props = regionprops(label1)
		
		#Координаты полок
		list_of_true_lab = list_of_shelves(props,row,col)
	 
		image = []
		flag=0
		a,b=0,0
		dist=0

		for i in range(len(list_of_true_lab)):	
			t,r=list_of_true_lab[i][0],list_of_true_lab[i][1]
			
			if r-a < 0.2*row:
				continue

			if r-t < 0.05*row:
				continue
			
			flag+=1	

			if r-t < 0.2*row:
				if dist!=0:
					if r-b < dist and r-b >= 0.1*row:
						end = saved[b:r:1,::1]
					else:
						if (r-b-dist)*0.5 > b-r+dist:
							end = saved[r-dist:r:1,::1]
						else:
							continue
				else:
					end = saved[:r:1,::1]
					dist = r
			else:	
				end = saved[t:r:1,::1]
				dist = r-t
			
			imsave('zdone/'+name+'_testeros'+str(i)+'.jpg',end)
								
			a,b=t,r
	
		if flag==0:
			img = img.astype("float32")/255

			#Инициализируем переменные
			rows = len(img)
			cols = len(img[0])

			#Полутоновое изображение
			gray = rgb2gray(img) 

			#Бинаризация изображения
			gray = binar(gray)

			#Разметка найденных объектов
			label1 = label(gray)

			#Получение данных для каждого из помеченных объектов
			props = regionprops(label1)
			
			#Координаты полок
			list_of_true_lab = list_of_shelves(props,rows,cols)

			image = []

			#Вырезаем полку и то, что на ней стоит
			if len(list_of_true_lab) > 1:
				
				dist = trunc((list_of_true_lab[1][0] - list_of_true_lab[0][1])*0.85)
				for i in range(len(list_of_true_lab)):
					print(list_of_true_lab[i][1] - list_of_true_lab[i][0])
					
					if list_of_true_lab[i][0]-dist > 0:
						end = saved[list_of_true_lab[i][0]-dist:list_of_true_lab[i][1]:1,::1]						
						
					elif list_of_true_lab[i][1] - list_of_true_lab[i][0] < 0.15 * rows:
						continue
					else:
						end = saved[:list_of_true_lab[i][1]:1,::1]
					imsave('zdone/'+name+'_testeros'+str(i)+'.jpg',end)


		print()


main()