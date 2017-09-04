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
parser.add_argument("-b","--BatchNum", type=int,
                    help="Number of Batches",
                    default = '100')
parser.add_argument("-c","--CheckP", type=str,
                    help="Checkpoint Dir",
                    default = None)
args = parser.parse_args()

NumBatch = args.BatchNum
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

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class Generator(object):
    def __init__(self, input_height=32, input_width=32, crop=True,
         batch_size=64, sample_num = 64, output_height=32, output_width=32,
         y_dim=None, z_dim=100, gf_dim=128, df_dim=128,
         gfc_dim=1024, dfc_dim=1024, c_dim=3,checkpoint_dir=None, dataset_name='default'):
        
        #self.sess = sess
        self.sess = tf.Session() 
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop = crop
        self.checkpoint_dir=checkpoint_dir
        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        #self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')

        self.dataset_name = dataset_name
        self.z = tf.placeholder(tf.float32, [1, self.z_dim], name='z')#J.L.
        
        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        with tf.variable_scope("generator") as scope:
            #scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            print(s_h)
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            print(s_h2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            print(s_h4)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            print(s_h8)

            # project `z` and reshape #J.L.
            self.z_, self.h0_w, self.h0_b = linear(self.z, self.gf_dim*4*s_h8*s_w8, 'g_h0_lin', with_w=True)
            self.h0 = tf.nn.relu(self.g_bn0(self.z_))
            print('h:')

            h0 = tf.reshape(self.h0, [-1, s_h8, s_w8, self.gf_dim * 4])
            print(h0.get_shape().as_list())
            
            self.h1, self.h1_w, self.h1_b = deconv2d_d2(h0, [self.batch_size, s_h4, s_w4, self.gf_dim*4],name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            print(h1.get_shape().as_list())

            h2, self.h2_w, self.h2_b = deconv2d_d1(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            print(h2.get_shape().as_list())

            h3, self.h3_w, self.h3_b = deconv2d_d2(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*2], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            print(h3.get_shape().as_list())

            h4, self.h4_w, self.h4_b = deconv2d_d1(h3, [self.batch_size, s_h2, s_w2, self.gf_dim],name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))
            print(h4.get_shape().as_list())

            h5, self.h5_w, self.h5_b = deconv2d_d2(h4, [self.batch_size, s_h, s_w, self.gf_dim], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))
            print(h5.get_shape().as_list())

            h6, self.h6_w, self.h6_b = deconv2d_d1(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)
            print(h6.get_shape().as_list())

        self.saver = tf.train.Saver()

        #def load(self):
        print(" [*] Reading checkpoints...")
        #checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            #return True
        else:
            print(" [*] Failed to find a checkpoint")
            exit(0)#return False
        
        self.genImage = tf.nn.tanh(h6)

    def __call__(self, z):
        return(self.sess.run(self.genImage, feed_dict={self.z: z}))
        #return tf.nn.tanh(h6)

#generator = Generator(checkpoint_dir=checkpoint_dir, dataset_name='default')
#prior = NormalPrior()
#kernel = ParsenDensityEstimator()
#model = ais.Model(generator, prior, kernel, 0.25, 10000)

PicPath = '../../DataImageNet/Image32/AllPx_valid.npy'
AllPx = np.load(PicPath)
# train32
# for i in xrange(9):
#    tmpPx = load_databatch(data_folder = PicPath, idx = i+2)
#    AllPx = np.vstack((AllPx, tmpPx))
# valid32
# AllPx = load_databatch(data_folder = PicPath, train = 'valid')

xx = AllPx[:64, :]
# Nxx = int(AllPx.shape[0]/64)
# for i in xrange(Nxx-1):
#   xx = AllPx[i*64:(i+1)*64, :]
#   blahblah...

# Get 100 f(x) values
#p = norm()
#x = np.linspace(norm.ppf(0.01, loc=3, scale=2), norm.ppf(0.99, loc=3, scale=2), 100)
#p1 = norm.pdf(x, loc=3, scale=2)
#xx = np.reshape(x, (100, 1))

schedule = ais.get_schedule(100, rad=4)
print(schedule)
#p2 = np.exp(model.ais(xx, schedule))
#print(p2)

#lld = model.ais(xx, schedule)
#print('lld_mean:')
#print(np.mean(lld))
#plt.plot(x, p1)
#plt.plot(x, p2)
#plt.show()
