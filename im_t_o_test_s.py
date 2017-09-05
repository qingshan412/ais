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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b","--SampleNum", type=int,
                    help="Number of Samples",
                    default = '64')
parser.add_argument("-c","--CheckP", type=str,
                    help="Checkpoint Dir",
                    default = None)
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
    def __init__(self, #sess, #input_height=32, input_width=32, crop=True,
         batch_size=64, sample_num = 64, output_height=32, output_width=32,
         y_dim=None, z_dim=100, gf_dim=128, df_dim=128,
         gfc_dim=1024, dfc_dim=1024, c_dim=3,checkpoint_dir=None, dataset_name='default'):
        
        self.input_dim = z_dim
        self.output_dim = 3*output_height*output_width
        self.sess = sess
        new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
        new_saver.restore(sess, 'my-save-dir/my-model-10000')
        #saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        self.v3h = tf.get_collection("v3")[0]
        self.a23th = tf.get_collection("tfadd")[0]

    def __call__(self, z):
        return self.sess.run(self.a23th, feed_dict={self.v3h:z})

#with tf.Session() as sess:
generator = Generator(tf.Session(), sample_num=NumSample)
print('init success!')
z = np.random.normal(0.0, 1.0, [64 * NumSample, 100])
ct = generator(z)
print(ct)
#prior = NormalPrior()
#kernel = ParsenDensityEstimator()
#model = ais.Model(generator, prior, kernel, 0.25, 10000)
print('try...')
exit(0)
