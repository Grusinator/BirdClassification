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
from lib.utils.image_reader import (
    RandomTransformer,
    SegmentationDataGenerator)

from lib.utils.SegDataGenerator import SegDataGenerator



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
# path = os.path.normpath(r"training_data/image_annotations_png/dag1")
#
# @click.command()
# @click.option('--train-dir', type=click.Path(exists=True),
#               default=os.path.normpath(r"training_data/image_annotations_png/dag1"))
# @click.option('--train-list-fname', type=click.Path(exists=True),
#               default=os.path.normpath(r"%s/train.txt"%path))
# @click.option('--val-list-fname', type=click.Path(exists=True),
#               default=os.path.normpath(r"%s/val.txt"%path))
# @click.option('--img-root', type=click.Path(exists=True),
#               default=os.path.normpath(r"%s/img/"%path))
# @click.option('--mask-root', type=click.Path(exists=True),
#               default=os.path.normpath(r"%s/mask/"%path))
# @click.option('--weights-path', type=click.Path(exists=True),
#               default= os.path.normpath(r"pretrained-models/dilation8_pascal_voc/dilation8_pascal_voc.npy"))#vgg_conv.npy')
#
# @click.option('--batch-size', type=int, default=1)
# @click.option('--learning-rate', type=float, default=1e-4)


@click.command()
@click.option('--train-data-path', type=click.Path(exists=True),
              default=os.path.normpath(r"training_data/image_annotations_png/dag2_better_all"))
@click.option('--pretrained-path', type=click.Path(exists=True),
              default= os.path.normpath(r"pretrained-models/dilation8_pascal_voc/dilation8_pascal_voc.npy"))#vgg_conv.npy')
@click.option('--weights-save-path', type=click.Path(exists=False),
              default= os.path.normpath(r"cnn-models/latest.hdf5"))#vgg_conv.npy')


@click.option('--batch-size', type=int, default=1)
@click.option('--epochs', type=int, default=15)
@click.option('--learning-rate', type=float, default=1e-4)

def train(train_data_path, pretrained_path ,weights_save_path,
          batch_size,
          epochs,
          learning_rate,
          target_size = (500,500),
          classes=2,
          loss_shape=None,
          label_file_ext='.png',
          data_file_ext='.jpg',
          ignore_label=255,
          label_cval=255):

    loss_shape = (target_size[0]*target_size[1]*classes,)

    #create full paths
    train_list_path = os.path.join(train_data_path, "train.txt")
    val_list_path = os.path.join(train_data_path, "val.txt")
    data_dir = os.path.join(train_data_path, "img")
    label_dir = os.path.join(train_data_path, "mask")


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
        monitor='val_loss',
        save_best_only=True,
        period=3
    )
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
        get_frontend(*target_size)) #(w,h))

    load_weights(model, pretrained_path)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                  metrics=['accuracy'])

    # Build absolute image paths
    def build_abs_paths(basenames):
        img_fnames = [os.path.join(data_dir, f) + data_file_ext for f in basenames]
        mask_fnames = [os.path.join(label_dir, f) + label_file_ext for f in basenames]
        return img_fnames, mask_fnames

    train_basenames = [l.strip() for l in open(train_list_path).readlines()]
    val_basenames = [l.strip() for l in open(val_list_path).readlines()][:500]

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
    

    # Create image generators for the training and validation sets. Validation has
    # no data augmentation.
    transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
    datagen_train = SegmentationDataGenerator(transformer_train)


    datagen_train2 = SegDataGenerator(zoom_range=2,#[0.5, 2.0],
                                     #zoom_maintain_shape=True,
                                     #crop_mode='center',
                                     crop_mode='random',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=True,
                                     #channel_shift_range=20.,
                                     fill_mode='constant',
                                     label_cval=label_cval)



    transformer_val = RandomTransformer(horizontal_flip=False, vertical_flip=False)
    datagen_val = SegmentationDataGenerator(transformer_val)




    
    skipped_report_cback = callbacks.LambdaCallback(
        on_epoch_end=lambda a, b: open(
            '{}/skipped.txt'.format(checkpoints_folder), 'a').write(
            '{}\n'.format(datagen_train.skipped_count)))

    #generator from keras-fcn
    generator2 = datagen_train2.flow_from_directory(
            file_path=train_list_path,
            data_dir=data_dir, data_suffix=data_file_ext,
            label_dir=label_dir, label_suffix=label_file_ext,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=ignore_label,
            # save_to_dir='Images/'
        )



    #generator from original 
    generator = datagen_train.flow_from_list(
            train_img_fnames,
            train_mask_fnames,
            shuffle=True,
            batch_size=batch_size,
            img_target_size=target_size,
            mask_target_size=(16, 16))


    validation_data = datagen_val.flow_from_list(
            val_img_fnames,
            val_mask_fnames,
            batch_size=5,
            img_target_size=target_size,
            mask_target_size=(16, 16))

    callback_list = [
            model_checkpoint,
            tensorboard_cback,
            csv_log_cback,
            reduce_lr_cback,
            skipped_report_cback,
        ]

    model.fit_generator(
        generator=generator,
        verbose=1,
        steps_per_epoch=len(train_img_fnames),
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=len(val_img_fnames),
        callbacks=callback_list)

    # model.fit_generator(
    #     datagen_train.flow_from_list(
    #         train_img_fnames,
    #         train_mask_fnames,
    #         shuffle=True,
    #         batch_size=batch_size,
    #         img_target_size=(w, h),
    #         mask_target_size=(16, 16)),
    #     verbose=1,
    #     steps_per_epoch=len(train_img_fnames),
    #     nb_epoch=epochs,
    #     validation_data=datagen_val.flow_from_list(
    #         val_img_fnames,
    #         val_mask_fnames,
    #         batch_size=5,
    #         img_target_size=(w, h),
    #         mask_target_size=(16, 16)),
    #     validation_steps=len(val_img_fnames),
    #     callbacks=[
    #         model_checkpoint,
    #         tensorboard_cback,
    #         csv_log_cback,
    #         reduce_lr_cback,
    #         skipped_report_cback,
    #     ])


    model.save_weights(weights_save_path)


if __name__ == '__main__':
    train()
