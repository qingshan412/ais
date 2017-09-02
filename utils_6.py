"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import os
import csv
import cPickle
import numpy as np
from glob import glob
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file
def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

  elif option == 5:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      np.random.seed(idx)
      z_sample = np.random.uniform(-1, 1, [config.batch_size, dcgan.z_dim])
      #for kdx, z in enumerate(z_sample):
      #  z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))    
  elif option == 6:#not in use again
    for idx in xrange(0, 10):
      data = glob(os.path.join("./..//data", config.dataset, config.input_fname_pattern))
      input_image=np.expand_dims(get_image(data[idx],
                        input_height=config.input_height,
                        input_width= config.input_width,
                        resize_height=config.output_height,
                        resize_width=config.output_width,
                        crop=config.crop,
                        grayscale=False),axis=0)
      
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: input_image}).tolist()

      with open('./feature_map.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(vector)

  elif option == 7:# split into 5 files

    pic_label=load_training_data()
    tnum = 10000
    label = pic_label[1][0:5*tnum]
    for idx in xrange(0, tnum):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./feature_map_50k_1.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))


    for idx in xrange(tnum, tnum*2):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./feature_map_50k_2.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))


    for idx in xrange(tnum*2, tnum*3):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./feature_map_50k_3.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))


    for idx in xrange(tnum*3, tnum*4):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./feature_map_50k_4.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))


    for idx in xrange(tnum*4, tnum*5):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./feature_map_50k_3.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))
        
  elif option == 8: # make test csv

    pic_label=load_test_data()
    tnum = 10000
    label = pic_label[1][0:tnum]
    for idx in xrange(0, tnum):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./test_feature_map.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))

  elif option == 9: #make train csv, one file

    pic_label=load_training_data()
    tnum = 50000
    label = pic_label[1][0:tnum]
    for idx in xrange(0, tnum):
      vector = sess.run(dcgan.classifier,feed_dict={dcgan.image1: [pic_label[0][idx]]})
      with open('./feature_map_50k.csv','a') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(np.concatenate((vector,[label[idx]]),axis=0))

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
def _get_file_path(filename=""):
    return os.path.join("./../data", "cifar-10-batches-py/", filename)
def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)
    
    with open(file_path, 'rb') as fo:
      data = cPickle.load(fo)
    return data
def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images
def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)
    return images, cls
def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls

def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls