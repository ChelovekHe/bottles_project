#Task
#Created by Vorobyev Ruslan
#2016
import glob
from scipy.ndimage.measurements import find_objects, label
from skimage import measure, exposure, filters
from numpy import where, shape, sort, zeros, percentile, array
from scipy import ndimage
from skimage.io import imread,imsave
from skimage.morphology import closing, rectangle
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.measure import regionprops
from skimage.filters import threshold_adaptive
from skimage.morphology import disk
from skimage.filters.rank import median
from math import trunc
import net


# Функция превращает изображение в полутоновое
def rgb2gray (rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Бинаризация
def binar (gray_img):

	gray_img[where(gray_img>=0.5)]=1
	gray_img[where(gray_img<0.5)]=0

	return gray_img

# Список полок
def list_of_shelves (props,rows,cols):
	
	list_of_true = []
	tmp = []

	for i in range(len(props)):
			
			t,y,r,w=props[i].bbox
			if w-y >= 0.9*cols and r-t <=0.3*rows:
				#print (i)
				tmp.append(t)
				tmp.append(r)
				list_of_true.append(tmp)
				tmp = []

	return list_of_true   

# Показать определенную метку
def show_me_lab (rows,cols,labelok,label_n):

	sp_lab = np.zeros((rows,cols), dtype='uint8')
	for i in range (rows):
		for j in range(cols):
			
			if (labelok[i][j]==label_n):
				sp_lab[i][j] = 255
			
			else:
				sp_lab[i][j] = 0

	imsave('test.jpg',sp_lab)	

# Скользящее окно для детекции бутылок
def generate_with_window (ends,windows=8):
	
	image = []
	width = ends.shape[1]
	step = trunc(width/(2*windows))
	for i in range(0, width, step):
		
		cur_img = ends[::1,i:i+step*2:1]
		imsave('bottles/test'+str(trunc(i/step))+'.jpg',cur_img)
		image.append(resize(np.array(cur_img),(64,64)).transpose(2,0,1))

	return image


def main ():	
	
	#Считываем изображение и приводим все значения к отрезку [0,1]
	text='4'
	f_name=text+'.jpg'
	img = imread(f_name)
	saved = img.copy()
	img = img.astype("float32")/255

	#Инициализируем переменные
	rows=len(img)
	cols=len(img[0])

	#Полутоновое изображение
	gray = rgb2gray(img) 

	#Бинаризация изображения
	gray = binar(gray)

	#Разметка найденных объектов
	label1,n = label(gray)

	#Получение данных для каждого из помеченных объектов
	props=regionprops(label1)
	
	#Координаты полок
	list_of_true_lab = list_of_shelves(props,rows,cols)

	#Вывести полку с указанным лейблом label_n
	#show_me_lab(rows,cols,label1,label_n=591)

	image = []

	#Вырезаем полку и то, что на ней стоит
	if len(list_of_true_lab) > 1:
		
		dist = list_of_true_lab[1][0] - list_of_true_lab[0][1]
		for i in range(len(list_of_true_lab)):
			
			if list_of_true_lab[i][0]-dist > 0:
				
				end = saved[list_of_true_lab[i][0]-dist:list_of_true_lab[i][1]:1,::1]
				image.append(generate_with_window(end))

			else:
				
				end = saved[:list_of_true_lab[i][1]:1,::1]
				image.append(generate_with_window(end))

			#imsave('test'+str(i)+'.jpg',end)
	else:

		gray = median(gray, disk(3))
		gray = threshold_adaptive(gray, 15, "mean")    
		gray = closing(gray, rectangle(1, 3))
		label1,n = label(gray)
		props=regionprops(label1)
		list_of_true_lab = list_of_shelves(props,rows,cols)
		
		if len(list_of_true_lab) > 1:
			
			dist = list_of_true_lab[1][0] - list_of_true_lab[0][1]
			for i in range(len(list_of_true_lab)):
			
				if list_of_true_lab[i][0]-dist > 0:
				
					end = saved[list_of_true_lab[i][0]-dist:list_of_true_lab[i][1]:1,::1]
					image.append(generate_with_window(end))

				else:
				
					end = saved[:list_of_true_lab[i][1]:1,::1]
					image.append(generate_with_window(end))

				#imsave('test'+str(i)+'.jpg',end)
		else:
			end = saved[::1,::1]
			image.append(generate_with_window(end))
	

	images=np.array(image)	

	#Подаем матрицу нейросети
	net.net(images)


main()