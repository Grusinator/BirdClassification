#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment images using weights from Fisher Yu (2016). Defaults to
settings for the Pascal VOC dataset.
'''

from __future__ import print_function, division

import argparse
import os
import functools

import numpy as np
from PIL import Image
from IPython import embed

from model import get_frontend, add_softmax, add_context
from utils import interp_map, pascal_palette


from lib.utils.image_splitter_merger import image_splitter_merger

# Settings for the Pascal dataset
input_width, input_height = 900, 900
label_margin = 186

has_context_module = False


def read_input_folder(folder):
    filelist = []
    for file in os.listdir(folder):
        if file.endswith(".png") | file.endswith(".jpg"):
            filelist.append(os.path.join(folder, file))
    return filelist

def get_base_filename(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]



def predict_from_folder(args):

    model = get_trained_model(args)

    input_list = read_input_folder(args.input_path)

    max_size = (500, 500)

    for input_image_path in input_list:

        print("predicting image: "+ input_image_path)

        ism = image_splitter_merger(max_size)

        # devide input image into suitable prediction sizes
        subimg_list = ism.image_splitter(Image.open(input_image_path))
        print("contains: %d subimages" %len(subimg_list))


        subimg_list = map(functools.partial(transform_image, mean=args.mean), subimg_list)

        # predict on each image
        annotated_subimg_list = map(functools.partial(predict_image,model=model), subimg_list)

        # merge the subsections
        annotated_image = ism.image_merger(map(toPILImage, annotated_subimg_list))
        # construct output path
        outputpath = os.path.join(args.output_path, get_base_filename(input_image_path) + ".jpg")
        #save image in output folder
        print('Saving results to: ', outputpath)
        with open(outputpath, 'wb') as out_file:
            annotated_image.save(out_file)

def toPILImage(array):
    return Image.fromarray(array.astype('uint8'), 'RGB')

def transform_image(image, mean = [0, 0, 0]):
    # Load image and swap RGB -> BGR to match the trained weights
    try:
        image_rgb = np.array(image).astype(np.float32)
    except TypeError as e:
        print("not valid type")

    image = image_rgb[:, :, ::-1] - mean
    return image

def predict_image(image, model):
    print(".", end=" ")
    image_size = image.shape

    # Network input shape (batch_size=1)
    net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin

    # This simplified prediction code is correct only if the output
    # size is large enough to cover the input without tiling
    assert image_size[0] < output_height
    assert image_size[1] < output_width

    # Center pad the original image by label_margin.
    # This initial pad adds the context required for the prediction
    # according to the preprocessing during training.
    image = np.pad(image,
                   ((label_margin, label_margin),
                    (label_margin, label_margin),
                    (0, 0)), 'reflect')

    # Add the remaining margin to fill the network input width. This
    # time the image is aligned to the upper left corner though.
    margins_h = (0, input_height - image.shape[0])
    margins_w = (0, input_width - image.shape[1])
    image = np.pad(image,
                   (margins_h,
                    margins_w,
                    (0, 0)), 'reflect')

    # Run inference
    net_in[0] = image
    prob = model.predict(net_in)[0]

    # Reshape to 2d here since the networks outputs a flat array per channel
    prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
    prob = prob.reshape((prob_edge, prob_edge, 21))

    # Upsample
    #if args.zoom > 1:
    #    prob = interp_map(prob, args.zoom, image_size[1], image_size[0])

    # Recover the most likely prediction (actual segment class)
    prediction = np.argmax(prob, axis=2)

    # Apply the color palette to the segmented image
    color_image = np.array(pascal_palette)[prediction.ravel()].reshape(
        prediction.shape + (3,))

    return color_image

def get_trained_model(args):
    """ Returns a model with loaded weights. """

    model = get_frontend(input_width, input_height)

    if has_context_module:
        model = add_context(model)

    model = add_softmax(model)

    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(args.weights_path, encoding='latin1').item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(args.weights_path)

    if args.weights_path.endswith('.npy'):
        load_tf_weights()
    elif args.weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    return model


def forward_pass(args):
    ''' Runs a forward pass to segment the image. '''

    model = get_trained_model(args)




    # Load image and swap RGB -> BGR to match the trained weights
    image_rgb = np.array(Image.open(args.input_path)).astype(np.float32)
    image = image_rgb[:, :, ::-1] - args.mean
    image_size = image.shape

    # Network input shape (batch_size=1)
    net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin

    # This simplified prediction code is correct only if the output
    # size is large enough to cover the input without tiling
    assert image_size[0] < output_height
    assert image_size[1] < output_width

    # Center pad the original image by label_margin.
    # This initial pad adds the context required for the prediction
    # according to the preprocessing during training.
    image = np.pad(image,
                   ((label_margin, label_margin),
                    (label_margin, label_margin),
                    (0, 0)), 'reflect')

    # Add the remaining margin to fill the network input width. This
    # time the image is aligned to the upper left corner though.
    margins_h = (0, input_height - image.shape[0])
    margins_w = (0, input_width - image.shape[1])
    image = np.pad(image,
                   (margins_h,
                    margins_w,
                    (0, 0)), 'reflect')

    # Run inference
    net_in[0] = image
    prob = model.predict(net_in)[0]

    # Reshape to 2d here since the networks outputs a flat array per channel
    prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
    prob = prob.reshape((prob_edge, prob_edge, 21))

    # Upsample
    if args.zoom > 1:
        prob = interp_map(prob, args.zoom, image_size[1], image_size[0])

    # Recover the most likely prediction (actual segment class)
    prediction = np.argmax(prob, axis=2)

    # Apply the color palette to the segmented image
    color_image = np.array(pascal_palette)[prediction.ravel()].reshape(
        prediction.shape + (3,))

    print('Saving results to: ', args.output_path)
    with open(args.output_path, 'wb') as out_file:
        Image.fromarray(color_image).save(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='?', default='validation_data/test1/input_folder',
                        help='Required path to input image folder')
    parser.add_argument('--output_path', default='validation_data/test1/results',
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=[98.63, 75.17, 23.57],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--zoom', default=8, type=int,
                        help='Upscaling factor')
    parser.add_argument('--weights_path', default='cnn-models/latest.hdf5', #'./dilation_pascal16.npy',
                        help='Weights file')

    args = parser.parse_args()



    if not args.output_path:
        dir_name, file_name = os.path.split(args.input_path)
        args.output_path = os.path.join(
            dir_name,
            '{}_seg.png'.format(
                os.path.splitext(file_name)[0]))

    predict_from_folder(args)


    #forward_pass(args)


if __name__ == "__main__":
    main()
