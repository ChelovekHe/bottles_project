import glob
from numpy import where, shape, zeros, dot, array
from skimage.io import imread, imsave
from skimage.morphology import closing, rectangle, disk
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.filters import threshold_adaptive
from skimage.filters.rank import median
from math import trunc
import cv2
import numpy as np
import os
import net

# Функция превращает изображение в полутоновое


def rgb2gray(rgb):

    return dot(rgb[..., :3], [0.299, 0.587, 0.114])

# Бинаризация


def binar(gray_img):

    gray_img[where(gray_img >= 0.5)] = 1
    gray_img[where(gray_img < 0.5)] = 0

    return gray_img


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def list_of_shelves(props, rows, cols):

    list_of_true = []
    tmp = []

    for i in range(len(props)):

        t, y, r, w = props[i].bbox
        if w - y >= 0.8 * cols:
            #print (i)
            tmp.append(t)
            tmp.append(r)
            list_of_true.append(tmp)
            tmp = []

    return list_of_true


def show_me_lab(rows, cols, labelok, label_n):

    sp_lab = zeros((rows, cols), dtype='uint8')
    for i in range(rows):
        for j in range(cols):

            if (labelok[i][j] == label_n):
                sp_lab[i][j] = 255

            else:
                sp_lab[i][j] = 0

    cv2.imwrite("label.jpg", sp_lab)


# Скользящее окно для детекции бутылок
def generate_with_window(ends, name, windows=20):

    image = []
    image1 = []
    width = ends.shape[1]
    step = trunc(width / (1 * windows))
    for i in range(0, width, step):

        cur_img = ends[::1, i:i + step * 1:1]
        imsave('bottles/test' + name +
               str(trunc(i / step)) + '_big.jpg', cur_img)
        image.append(resize(array(cur_img), (64, 64)).transpose(2, 0, 1))

    step = trunc(width / (4 * windows))
    for i in range(0, width, step):

        cur_img = ends[::1, i:i + step * 2:1]
        # imsave('bottles/test'+name+str(trunc(i/step))+'_small.jpg',cur_img)
        image1.append(resize(array(cur_img), (64, 64)).transpose(2, 0, 1))

    return image, image1


def first_try_to_detect(image, saved, name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gray_x = cv2.Sobel(blurred, ddepth=-1, dx=1, dy=0)
    gray_y = cv2.Sobel(blurred, ddepth=-1, dx=0, dy=1)

    row = gray_x.shape[0]
    col = gray_x.shape[1]

    a = np.zeros((row, col), dtype='int')
    a += 2
    gray = gray_x**a + gray_y**a
    a = a.astype("float32") - 1.5
    gray = np.trunc(gray ** a).astype("int")

    middle = gray.sum() / (row * col)

    gray[np.where(gray < middle)] = 0
    gray[np.where(gray >= middle)] = 255

    label1 = label(gray, neighbors=8)

    # Получение данных для каждого из помеченных объектов
    props = regionprops(label1)

    # Координаты полок
    list_of_true_lab = list_of_shelves(props, row, col)

    image = []
    image1 = []
    flag = 0
    a, b = 0, 0
    dist = 0

    for i in range(len(list_of_true_lab)):
        t, r = list_of_true_lab[i][0], list_of_true_lab[i][1]

        if r - a < 0.2 * row:
            continue

        if r - t < 0.05 * row:
            continue

        flag += 1

        if r - t < 0.2 * row:
            if dist != 0:
                if r - b < dist and r - b >= 0.1 * row:
                    end = saved[b:r:1, ::1]
                else:
                    if (r - b - dist) * 0.5 > b - r + dist:
                        end = saved[r - dist:r:1, ::1]
                    else:
                        continue
            else:
                end = saved[:r:1, ::1]
                dist = r
        else:
            end = saved[t:r:1, ::1]
            dist = r - t
        cur, cur1 = generate_with_window(end, name)
        image.append(cur)
        image1.append(cur1)
        imsave('zdone/' + name + '_testeros' + str(i) + '.jpg', end)

        a, b = t, r

    return image, image1, flag


def second_try_to_detect(img, saved, name):
    img = img.astype("float32") / 255

    # Инициализируем переменные
    rows = len(img)
    cols = len(img[0])

    # Полутоновое изображение
    gray = rgb2gray(img)

    # Бинаризация изображения
    gray = binar(gray)

    # Разметка найденных объектов
    label1 = label(gray)

    # Получение данных для каждого из помеченных объектов
    props = regionprops(label1)

    # Координаты полок
    list_of_true_lab = list_of_shelves(props, rows, cols)

    image = []

    # Вырезаем полку и то, что на ней стоит
    if len(list_of_true_lab) > 1:

        dist = trunc((list_of_true_lab[1][0] - list_of_true_lab[0][1]) * 0.85)
        for i in range(len(list_of_true_lab)):

            if list_of_true_lab[i][0] - dist > 0:
                end = saved[
                    list_of_true_lab[i][0] -
                    dist:list_of_true_lab[i][1]:1,
                    ::1]

            elif list_of_true_lab[i][1] - list_of_true_lab[i][0] < 0.15 * rows:
                continue
            else:
                end = saved[:list_of_true_lab[i][1]:1, ::1]

            cur, cur1 = generate_with_window(end, name)
            image.append(cur)
            image1.append(cur1)
            imsave('zdone/' + name + '_testeros' + str(i) + '.jpg', end)

    return image, image1


def maximum(list_of_proba, list_of_class):
    k = 0
    flag = 1
    maxima = 0
    index_list = []
    # print(len(list_of_proba))
    # print(len(list_of_class))
    print(list_of_proba[0])
    for i in range(len(list_of_proba)):
        if list_of_proba[i][list_of_class[i]] >= maxima:
            flag = 1
        elif flag == 1:
            k += 1
            flag = 0
            index_list.append(i - 1)
        maxima = list_of_proba[i][list_of_class[i]]
        print(maxima)

    print(k)
    print(index_list)


def print_class(list_of_class):
    for i in range(len(list_of_class)):
        if list_of_class[i] == 0:
            print("zolotaya", end=" ")
        elif list_of_class[i] == 1:
            print("karolina", end=" ")
        elif list_of_class[i] == 2:
            print("blago", end=" ")
        elif list_of_class[i] == 3:
            print("anninskoe", end=" ")
        elif list_of_class[i] == 4:
            print("krasnaya_niz", end=" ")
        elif list_of_class[i] == 5:
            print("ideal", end=" ")
        elif list_of_class[i] == 6:
            print("selyanochka", end=" ")
        elif list_of_class[i] == 7:
            print("oleyna", end=" ")
        elif list_of_class[i] == 8:
            print("sloboda", end=" ")
        elif list_of_class[i] == 9:
            print("altero", end=" ")
    print()


def main():
    key_word = 'testdata' + '/'
    file_path1 = '/home/ruslan/prac/planogramm/' + key_word
    for name in os.listdir(file_path1):
        if name == 'else':
            continue
        print(name)
        image = cv2.imread(key_word + name)
        img = imread(key_word + name)
        saved = img.copy()

        image, image1, flag = first_try_to_detect(image, saved, name)

        if flag == 0:
            image, image1 = second_try_to_detect(img, saved, name)

        images = array(image)
        images1 = array(image1)

        # Подаем матрицу нейросети
        list_of_list_of_class, list_of_list_of_proba = net.net(images, mode=1)

        for ind in range(len(list_of_list_of_class)):
            list_of_proba = list_of_list_of_proba[ind]
            list_of_class = list_of_list_of_class[ind]
            print_class(list_of_class)
            maximum(list_of_proba, list_of_class)

main()
