#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment images using weights from Fisher Yu (2016). Defaults to
settings for the Pascal VOC dataset.
'''

from __future__ import print_function, division

import argparse
import os
from progress.bar import Bar

import numpy as np
from PIL import Image
from IPython import embed

from model import get_frontend, add_softmax, add_context
from lib.utils import interp_map, pascal_palette


from lib.utils.image_splitter_merger import image_splitter_merger

# Settings for the Pascal dataset
input_width, input_height = 900, 900
label_margin = 186

has_context_module = False

def get_trained_model(weights_path):
    """ Returns a model with loaded weights. """

    model = get_frontend(input_width, input_height)

    if has_context_module:
        model = add_context(model)

    model = add_softmax(model)

    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(weights_path, encoding='latin1').item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(weights_path)

    if weights_path.endswith('.npy'):
        load_tf_weights()
    elif weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    return model


def transform_image(image, mean = [0, 0, 0]):
    # Load image and swap RGB -> BGR to match the trained weights
    try:
        image_rgb = np.array(image).astype(np.float32)
    except TypeError as e:
        print("not valid type")
        return

    image = image_rgb[:, :, ::-1] - mean
    return image

def predict_image(image, model,pgbar = None):
    print(".", end=" ")
    image_size = image.shape

    if pgbar != None:
        pgbar.next()

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

    args_zoom = 8
    #Upsample
    if args_zoom > 1:
       prob = interp_map(prob, args_zoom, image_size[1], image_size[0])

    # Recover the most likely prediction (actual segment class)
    prediction = np.argmax(prob, axis=2)

    # Apply the color palette to the segmented image
    color_image = np.array(pascal_palette)[prediction.ravel()].reshape(
        prediction.shape + (3,))

    return color_image

def predict_single_image(input_path, output_path, model, mean, input_size):

    ism = image_splitter_merger(input_size)

    # devide input image into suitable prediction sizes
    subimg_list = ism.image_splitter(Image.open(input_path))

    trans_subimg_list = [transform_image(subimg, mean=mean) for subimg in subimg_list]

    bar = Bar('Processing', max=len(subimg_list))

    # predict on each image
    annotatedimg_list = [predict_image(subimg,model=model,pgbar=bar) for subimg in trans_subimg_list]
    bar.finish()
    #merge to one image again
    annotated_image = ism.image_merger(annotatedimg_list)

    if not output_path:
        dir_name, file_name = os.path.split(input_path)
        output_path = os.path.join(
            dir_name,
            '{}_seg.png'.format(
                os.path.splitext(file_name)[0]))
    elif os.path.isdir(output_path):
        dir_name, file_name = os.path.split(input_path)
        output_path = os.path.join(
            output_path,
            '{}_seg.png'.format(
                os.path.splitext(file_name)[0]))
    elif output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        pass
    else:
        print("something is wrong here fix...")

    #save image in output folder
    print('Saving results to: ', output_path)
    with open(output_path, 'wb') as out_file:
        annotated_image.save(out_file)


def toPILImage(array):
    return Image.fromarray(array.astype('uint8'), 'RGB')


def read_input_folder(folder):
    filelist = []
    for file in os.listdir(folder):
        if file.endswith(".png") | file.endswith(".jpg"):
            filelist.append(os.path.join(folder, file))
    return filelist

def get_base_filename(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def predict_from_folder(input_path, output_path, model, mean, input_size):

    input_list = read_input_folder(input_path)

    if output_path is None:
        output_path = input_path

    for input_image_path in input_list:

        print("predicting image: "+ get_base_filename(input_image_path))

        output_image_path = os.path.join(output_path, get_base_filename(input_image_path) + "_seg.png")

        predict_single_image(input_image_path, output_image_path,model,mean, input_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='?', default='evaluation/input/',#296_before_crop_double.jpg',
                        help='Required path to input image')
    parser.add_argument('--output_path', default='evaluation/output/',
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=\
                        [98.63, 75.17, 23.57], #birds
                        #[102.93, 111.36, 116.52], #PASCAL
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    # parser.add_argument('--zoom', default=8, type=int,
    #                     help='Upscaling factor')
    parser.add_argument('--weights_path', default='cnn-models/latest.hdf5',
                        #'cnn-models/ep10-vl0.0908.hdf5',
                        # #'./dilation_pascal16.npy',
                        help='Weights file')
    parser.add_argument('--input_size', default=(500,500),
                        help='max input size of classifier')

    args = parser.parse_args()

    model = get_trained_model(args.weights_path)

    if os.path.isfile(args.input_path):
        predict_single_image(args.input_path, args.output_path, model, args.mean, args.input_size)
    elif os.path.isdir(args.input_path):
        predict_from_folder(args.input_path, args.output_path, model, args.mean, args.input_size)
    else:
        print("Does it exist?  Is it a file, or a directory?")

    print("done!")


if __name__ == "__main__":
    main()
