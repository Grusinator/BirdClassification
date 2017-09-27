

import xml.etree.ElementTree
import os

from PIL import Image
import numpy as np

from skimage import filters
from skimage import data
from sklearn.model_selection import train_test_split
import random


def xmllabel2img(xmlfilenames, imagefilenames, img_size=(500,500,3)):
    imagelist = []

    if isinstance(xmlfilenames, list):

        for xmlfilename in xmlfilenames:
            filename = remove_path_and_ext(xmlfilename)

            if filename in map(remove_path_and_ext, imagefilenames):
                imagefilename = filter(lambda x: remove_path_and_ext(x) == filename, imagefilenames).__next__()
                imagelist.append(singlexmllabel2img(xmlfilename, imagefilename))
    else:
        imagelist.append(singlexmllabel2img(xmlfilenames, imagefilenames))

    return imagelist
    
def singlexmllabel2img(xmlfilename, imagefilename, img_size=(500,500,3)):
    e = xml.etree.ElementTree.parse(xmlfilename).getroot()
    obj = e.find('object')
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    xmax = int(bbox.find('xmax').text)
    ymin = int(bbox.find('ymin').text)
    ymax = int(bbox.find('ymax').text)

    path = e.find('path')



    

    xdif = xmax - xmin
    ydif = ymax - ymin
    xmean = np.round(np.mean([xmax, xmin]))
    ymean = np.round(np.mean([ymax, ymin]))

    x_delta = np.round(img_size[0]/2)
    y_delta = np.round(img_size[1]/2)

    #min max with specified image size
    xmin_f = xmean - x_delta
    xmax_f = xmin_f + img_size[0]
    ymin_f = ymean - y_delta
    ymax_f = ymin_f + img_size[1]




    im = Image.open(imagefilename)

    crop_rectangle_f = (xmin_f, ymin_f, xmax_f, ymax_f)
    cropped_im = im.crop(crop_rectangle_f)
    #cropped_im.show()

    crop_rectangle_inner = (xmin, ymin, xmax, ymax)

    xmin_b = int(xmin - xmin_f)
    xmax_b = int(img_size[0] - (xmax_f - xmax))
    ymin_b = int(ymin - ymin_f)
    ymax_b = int(img_size[1] - (ymax_f - ymax)) 

    data = np.zeros((img_size[0],img_size[1]), dtype=np.uint8)
    data[xmin_b:xmax_b, ymin_b:ymax_b] = 1
    mask_img = Image.fromarray(data)
    #img.show()

    crop_rectangle_b = (xmin_b, ymin_b, xmax_b, ymax_b)


    return cropped_im, mask_img


def remove_path_and_ext(path):
    base=os.path.basename(path)
    return os.path.splitext(base)[0]




#path = "smb://allkfs09/home/Projects/LiDAR_Bird_detection_2015/Hornsea/4_results/test/annotations/"

def getfilenames(path, ext):
    filenamelist = []

    for file in os.listdir(path):
        if file.endswith(ext):
            p = os.path.join(path, file)
            print(p)
            filenamelist.append(p)
    return filenamelist

def PIL2array(img):
    if img.mode == 'RGB':
        dim3 = 3
    elif img.mode == 'L':
        dim3 = 1
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], dim3)




xmlpath = '/home/wsh/python/xmllabel2img/dag1/annotations/'
imgpath = '/home/wsh/python/xmllabel2img/dag1/Images/'

imgfilenames = getfilenames(imgpath, '.jpg')
xmlfilenames = getfilenames(xmlpath, '.xml')


imagelist = xmllabel2img(xmlfilenames, imgfilenames)

#write to disk
outputpath = '/home/wsh/python/xmllabel2img/dag1/output/'
img_folder = 'img'




for img, xmlfile in zip(imagelist, xmlfilenames):
    filename = remove_path_and_ext(xmlfile)
    img[0].save('%s/img/%s.jpg' %(outputpath, filename))

    #convert to grayscale and truncate border
    #trunc = PIL2array(img[0].convert('L')) #* PIL2array(img[1])

    #gray = img[0].convert('L')
    #print(trunc)

    #test = np.array(range(100),dtype=np.double)
    #val = filters.threshold_otsu(test)

    #mask = trunc < val
    #mask_img = Image.fromarray(mask)
    
    #mask_img.save('output_data/mask_filter/%s.jpg' %filename)

    img[1].save('%s/mask/%s.png' %(outputpath, filename))






#train,_,val,_ = train_test_split(xmlfilenames,xmlfilenames size=0.2)


random.shuffle(xmlfilenames)

nr = int(len(xmlfilenames) * 0.8)

train = xmlfilenames[0:nr]
val = xmlfilenames[nr:]

f_train = open('%s/train.txt'%outputpath, 'w')
for i in train:
    f_train.write(remove_path_and_ext(i) + '\n')  # python will convert \n to os.linesep
f_train.close()

f_val = open('%s/val.txt'%outputpath, 'w')
for i in val:
    f_val.write(remove_path_and_ext(i) +'\n')  # python will convert \n to os.linesep
f_val.close()