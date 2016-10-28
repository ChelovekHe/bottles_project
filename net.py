from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.optimizers import SGD
import img_to_nparr


def net (X_test, mode=1):

    img_width, img_height = 64, 64

    train_data_dir = 'datas2/train'
    validation_data_dir = 'datas2/validation'
    nb_train_samples = 18000
    nb_validation_samples = 40
    nb_epoch = 5


    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32)

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32)

    if mode == 0:
        model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples)

        model.save_weights('weights1.h5')
    else:
        model.load_weights('weights1.h5')

    list_of_list_of_class = []
    list_of_list_of_proba = []
    
    for ind in range(len(X_test)):
        list_of_class = model.predict_classes(X_test[ind], verbose = 0)
        list_of_proba = model.predict_proba(X_test[ind], verbose = 0)
        list_of_list_of_class.append(list_of_class)
        list_of_list_of_proba.append(list_of_proba)
    
    return list_of_list_of_class, list_of_list_of_proba

if __name__ == '__main__':
	X_test = img_to_nparr.img_to_nparr('/home/ruslan/datas2/validation/5')
	net(X_test)
