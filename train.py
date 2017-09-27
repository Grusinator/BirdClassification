#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil

import click
import numpy as np
from keras import callbacks, optimizers
from IPython import embed

from model import get_frontend, add_softmax
from utils.image_reader import (
    RandomTransformer,
    SegmentationDataGenerator)

h = 500
w = 500


def load_weights(model, weights_path):
    weights_data = np.load(weights_path, encoding='latin1').item()

    for layer in model.layers:
        if layer.name in weights_data.keys():
            layer_weights = weights_data[layer.name]
            layer.set_weights((layer_weights['weights'],
                               layer_weights['biases']))


"""
@click.command()
@click.option('--train-list-fname', type=click.Path(exists=True),
              default='/home/wsh/python/segmentation_keras/data/train.txt')
@click.option('--val-list-fname', type=click.Path(exists=True),
              default='/home/wsh/python/segmentation_keras/data/val.txt')
@click.option('--img-root', type=click.Path(exists=True),
              default='/home/wsh/python/segmentation_keras/data/img/')
@click.option('--mask-root', type=click.Path(exists=True),
              default='/home/wsh/python/segmentation_keras/data/mask/')
@click.option('--weights-path', type=click.Path(exists=True),
              default='conversion/converted/dilation8_pascal_voc.npy')#vgg_conv.npy')
@click.option('--batch-size', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-4)
"""

#training data path
#path = '/home/wsh/python/xmllabel2img/dag1/output/'
path = os.path.normpath(r"training_data/image_annotations_png/dag1")

@click.command()
@click.option('--train-list-fname', type=click.Path(exists=True),
              default=os.path.normpath(r"%s/train.txt"%path))
@click.option('--val-list-fname', type=click.Path(exists=True),
              default=os.path.normpath(r"%s/val.txt"%path))
@click.option('--img-root', type=click.Path(exists=True),
              default=os.path.normpath(r"%s/img/"%path))
@click.option('--mask-root', type=click.Path(exists=True),
              default=os.path.normpath(r"%s/mask/"%path))
@click.option('--weights-path', type=click.Path(exists=True),
              default= os.path.normpath(r"cnn-models/pretrained-models/dilation8_pascal_voc/dilation8_pascal_voc.npy"))#vgg_conv.npy')

@click.option('--batch-size', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-4)

def train(train_list_fname,
          val_list_fname,
          img_root,
          mask_root,
          weights_path,
          batch_size,
          learning_rate):

    # Create image generators for the training and validation sets. Validation has
    # no data augmentation.
    transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
    datagen_train = SegmentationDataGenerator(transformer_train)

    transformer_val = RandomTransformer(horizontal_flip=False, vertical_flip=False)
    datagen_val = SegmentationDataGenerator(transformer_val)

    train_desc = '{}-lr{:.0e}-bs{:03d}'.format(
        time.strftime("%Y-%m-%d %H.%M"),
        learning_rate,
        batch_size)
    checkpoints_folder = 'cnn-models/' + train_desc
    try:
        os.makedirs(checkpoints_folder)
    except OSError:
        shutil.rmtree(checkpoints_folder, ignore_errors=True)
        os.makedirs(checkpoints_folder)

    model_checkpoint = callbacks.ModelCheckpoint(
        checkpoints_folder + '/ep{epoch:02d}-vl{val_loss:.4f}.hdf5',
        monitor='loss')
    tensorboard_cback = callbacks.TensorBoard(
        log_dir='{}/tboard'.format(checkpoints_folder),
        histogram_freq=0,
        write_graph=False,
        write_images=False)
    csv_log_cback = callbacks.CSVLogger(
        '{}/history.log'.format(checkpoints_folder))
    reduce_lr_cback = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        verbose=1,
        min_lr=0.05 * learning_rate)

    model = add_softmax(
        get_frontend(w, h))

    load_weights(model, weights_path)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                  metrics=['accuracy'])

    # Build absolute image paths
    def build_abs_paths(basenames):
        img_fnames = [os.path.join(img_root, f) + '.jpg' for f in basenames]
        mask_fnames = [os.path.join(mask_root, f) + '.png' for f in basenames]
        return img_fnames, mask_fnames

    train_basenames = [l.strip() for l in open(train_list_fname).readlines()]
    val_basenames = [l.strip() for l in open(val_list_fname).readlines()][:500]

    train_img_fnames, train_mask_fnames = build_abs_paths(train_basenames)
    val_img_fnames, val_mask_fnames = build_abs_paths(val_basenames)


    # for fnames in train_img_fnames:
    #     if not os.path.exists(fnames):
    #         train_img_fnames.remove(fnames)
    #         print("removed: " + fnames)
    #
    # for fnames in val_img_fnames:
    #     if not os.path.exists(fnames):
    #         val_img_fnames.remove(fnames)
    #         print("removed: " + fnames)


    skipped_report_cback = callbacks.LambdaCallback(
        on_epoch_end=lambda a, b: open(
            '{}/skipped.txt'.format(checkpoints_folder), 'a').write(
            '{}\n'.format(datagen_train.skipped_count)))

    print(batch_size)
    print(len(train_img_fnames))

    model.fit_generator(
        datagen_train.flow_from_list(
            train_img_fnames,
            train_mask_fnames,
            shuffle=True,
            batch_size=batch_size,
            img_target_size=(w, h),
            mask_target_size=(16, 16)),
        verbose=2,
        steps_per_epoch=len(train_img_fnames),
        nb_epoch=40,
        validation_data=datagen_val.flow_from_list(
            val_img_fnames,
            val_mask_fnames,
            batch_size=2,
            img_target_size=(w, h),
            mask_target_size=(16, 16)),
        validation_steps=len(val_img_fnames),
        callbacks=[
            model_checkpoint,
            tensorboard_cback,
            csv_log_cback,
            reduce_lr_cback,
            skipped_report_cback,
        ])

    model.save_weights()


if __name__ == '__main__':
    train()
