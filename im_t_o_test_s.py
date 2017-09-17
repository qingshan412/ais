import tensorflow as tf
import numpy as np
import ais
import matplotlib.pyplot as plt
from priors import NormalPrior
from kernels import ParsenDensityEstimator
from scipy.stats import norm

#import re
#from ops_dcgan import *
from ops import *
import math
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b","--SampleNum", type=int,
                    help="Number of Samples",
                    default = '64')
parser.add_argument("-c","--CheckP", type=str,
                    help="Checkpoint Dir",
                    default = "../../DCGAN/original_storeMore/checkpoint/original24")
args = parser.parse_args()

NumSample = args.SampleNum
checkpoint_dir = args.CheckP

#class Generator(object):
#    def __init__(self, input_dim, output_dim):
#        self.input_dim = input_dim
#        self.output_dim = output_dim
#
#    def __call__(self, z):
#        return z * 2 + 3
#
#generator = Generator(1, 1)
#prior = NormalPrior()
#kernel = ParsenDensityEstimator()
#model = ais.Model(generator, prior, kernel, 0.25, 10000)

class Generator(object):
    def __init__(self, sess, #input_height=32, input_width=32, crop=True,
         batch_size=64, sample_num = 64, output_height=32, output_width=32,
         y_dim=None, z_dim=100, gf_dim=128, df_dim=128,
         gfc_dim=1024, dfc_dim=1024, c_dim=3,checkpoint_dir=None, dataset_name='default'):
        
        self.input_dim = z_dim
        self.output_dim = 3*output_height*output_width
        self.sess = sess
        new_saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir,'my-model-10000.meta'))
        new_saver.restore(sess, os.path.join(checkpoint_dir,'my-model-10000'))
        #saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        self.v3h = tf.get_collection("zr")[0]
        print("zr restored.")
        print(tf.shape(self.v3h))
        self.a23th = tf.get_collection("gen_op")[0]
        print("generator restored.")

    def __call__(self, z):
        print("get called!")
        return self.sess.run(self.a23th, feed_dict={self.v3h:z})

#with tf.Session() as sess:
generator = Generator(sess=tf.Session(), sample_num=NumSample, checkpoint_dir=checkpoint_dir)
print('init success!')#64 * NumSample
z = np.random.normal(0.0, 1.0, [NumSample, 100])
ct = generator(z)
print(ct)
#prior = NormalPrior()
#kernel = ParsenDensityEstimator()
#model = ais.Model(generator, prior, kernel, 0.25, 10000)
print('try...')
exit(0)
