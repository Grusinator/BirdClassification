#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

from lib.utils.SegDataGenerator import SegDataGenerator



from lib.utils.image_reader import (
    RandomTransformer,
    SegmentationDataGenerator)

from lib.utils.SegDataGenerator import SegDataGenerator







@click.command()
@click.option('--train-data-path', type=click.Path(exists=True),
              default=os.path.normpath(r"../training_data/image_annotations_png/dag2_better_all"))
@click.option('--pretrained-path', type=click.Path(exists=True),
              default=os.path.normpath(
                  r"../pretrained-models/dilation8_pascal_voc/dilation8_pascal_voc.npy"))  # vgg_conv.npy')
@click.option('--weights-save-path', type=click.Path(exists=False),
              default=os.path.normpath(r"cnn-models/latest.hdf5"))  # vgg_conv.npy')
@click.option('--batch-size', type=int, default=1)
@click.option('--epochs', type=int, default=15)
@click.option('--learning-rate', type=float, default=1e-4)
def train(train_data_path, pretrained_path, weights_save_path,
          batch_size,
          epochs,
          learning_rate,
          target_size=(500, 500),
          classes=2,
          loss_shape=None,
          label_file_ext='.png',
          data_file_ext='.jpg',
          ignore_label=255,
          label_cval=255):
    #loss_shape = (target_size[0] * target_size[1] * classes,)

    # create full paths
    train_list_path = os.path.join(train_data_path, "train.txt")
    val_list_path = os.path.join(train_data_path, "val.txt")
    data_dir = os.path.join(train_data_path, "img")
    label_dir = os.path.join(train_data_path, "mask")



    # Build absolute image paths
    def build_abs_paths(basenames):
        img_fnames = [os.path.join(data_dir, f) + data_file_ext for f in basenames]
        mask_fnames = [os.path.join(label_dir, f) + label_file_ext for f in basenames]
        return img_fnames, mask_fnames

    train_basenames = [l.strip() for l in open(train_list_path).readlines()]
    val_basenames = [l.strip() for l in open(val_list_path).readlines()][:500]

    train_img_fnames, train_mask_fnames = build_abs_paths(train_basenames)
    val_img_fnames, val_mask_fnames = build_abs_paths(val_basenames)



    transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
    datagen_train = SegmentationDataGenerator(transformer_train)

    generator = datagen_train.flow_from_list(
            train_img_fnames,
            train_mask_fnames,
            shuffle=True,
            batch_size=batch_size,
            img_target_size=target_size,
            mask_target_size=(16, 16))


    datagen_train2 = SegDataGenerator(zoom_range=[0.5, 2.0],
                                      zoom_maintain_shape=True,
                                      #crop_mode='center',
                                      #crop_mode='random',
                                      #crop_size=target_size,
                                      #pad_size=(505, 505),
                                      rotation_range=0.,
                                      shear_range=0,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      # channel_shift_range=20.,
                                      fill_mode='constant',
                                      label_cval=label_cval,
    )

    # generator from keras-fcn
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

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    img = label = None
    for data in generator:
        print(str(type(data[0])) + " " + str(data[0].shape))
        print(str(type(data[1])) + " " + str(data[1].shape))


        img_data = np.squeeze(data[0][0:,:,:,:])
        #label_data = data[1][0,:,:,:]

        print(img_data.shape)

        if img is None:
            img = plt.imshow(img_data)
        else:
            img.set_data(img_data)

        # if label is None:
        #     label = plt.imshow(label_data)
        # else:
        #     label.set_data(label_data)

        plt.pause(1)
        plt.draw()




if __name__ == '__main__':
    train()
