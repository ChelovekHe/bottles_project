from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

folder_num = 5
folder_name = 'datas2/train/'
data_size = 100

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')
for j in range(20):
    i = 0
    img = load_img(folder_name + str(folder_num) + '/' + str(j+1) + '.jpg') 
    x = img_to_array(img) 
    x = x.reshape((1,) + x.shape) 
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=folder_name + str(folder_num), save_format='jpeg'):
        i += 1
        if i >= data_size:
            break  
