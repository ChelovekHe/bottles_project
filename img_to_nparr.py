from skimage.io import imread,imsave
from skimage.transform import resize
import os
import numpy as np

def img_to_nparr(file_path1):	
	
	images=[]

	for name in os.listdir(file_path1):
		f_name=file_path1+"/"+name
		images.append(resize(np.array(imread(f_name)),(64,64)).transpose(2,0,1))

	images1=np.array(images)
	return  images1

if __name__ == '__main__':
	file_path1="/home/ruslan/Загрузки/archive/datas"
	img_to_nparr(file_path1)