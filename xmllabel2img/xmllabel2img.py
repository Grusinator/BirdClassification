
import xml.etree.ElementTree
import os
from PIL import Image
import numpy as np
import random
import click


@click.command()
@click.option('--labelpath', type=click.Path(exists=True),
              default=os.path.normpath(r"../training_data/image_annotations_xml/dag2_better_all/annotations/"))
@click.option('--imagepath', type=click.Path(exists=True),
              default=os.path.normpath(r"../training_data/image_annotations_xml/dag2_better_all/images/"))
@click.option('--outputpath', type=click.Path(exists=True),
              default=os.path.normpath(r"../training_data/image_annotations_png/dag2_better_all/"))
def process(labelpath,imagepath,outputpath):
    imgfilenames = getfilenames(imagepath, '.jpg')
    xmlfilenames = getfilenames(labelpath, '.xml')

    imagelist = xmllabel2img(xmlfilenames, imgfilenames)

    dirs = ['img', 'mask']

    full_dir = list(map(lambda x:os.path.join(outputpath,x), dirs))

    [create_if_not_exists(folder) for folder in full_dir]

    for img, xmlfile in zip(imagelist, xmlfilenames):
        filename = remove_path_and_ext(xmlfile)
        imgpath = os.path.join(outputpath, dirs[0], filename + '.jpg')
        maskpath = os.path.join(outputpath, dirs[1], filename + '.png')
        img[0].save(imgpath)
        img[1].save(maskpath)

    create_train_val_list(labelpath,outputpath)


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

def create_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)



def create_train_val_list(xmlpath,outputpath):
    xmlfilenames = getfilenames(xmlpath, '.xml')

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

if __name__ == "__main__":
    process()
