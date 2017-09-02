import tensorflow as tf
import numpy as np
import ais
import matplotlib.pyplot as plt
from priors import NormalPrior
from kernels import ParsenDensityEstimator
from scipy.stats import norm

#import re
#from ops import *
#from utils import *
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
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
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
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.saver = tf.train.Saver()

        #def load(self):
        print(" [*] Reading checkpoints...")
        #checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            #return True
        else:
            print(" [*] Failed to find a checkpoint")
            exit(0)#return False
        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*4*s_h8*s_w8, 'g_h0_lin', with_w=True)
            self.h0 = tf.nn.relu(self.g_bn0(self.z_))

            h0 = tf.reshape(self.h0, [-1, s_h8, s_w8, self.gf_dim * 4])
            
            self.h1, self.h1_w, self.h1_b = deconv2d_d2(h0, [self.batch_size, s_h4, s_w4, self.gf_dim*4],name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d_d1(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d_d2(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*2], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d_d1(h3, [self.batch_size, s_h2, s_w2, self.gf_dim],name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))

            h5, self.h5_w, self.h5_b = deconv2d_d2(h4, [self.batch_size, s_h, s_w, self.gf_dim], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))

            h6, self.h6_w, self.h6_b = deconv2d_d1(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)

    def __call__(self, z):
            return tf.nn.tanh(h6)


generator = Generator(checkpoint_dir=checkpoint_dir, dataset_name='default')
         #sess, input_height=32, input_width=32, crop=True,
         #batch_size=64, sample_num = 64, output_height=64, output_width=64,
         #y_dim=None, z_dim=100, gf_dim=128, df_dim=128,
         #gfc_dim=1024, dfc_dim=1024, c_dim=3,checkpoint_dir=checkpoint_dir, dataset_name='default')
prior = NormalPrior()
kernel = ParsenDensityEstimator()
model = ais.Model(generator, prior, kernel, 0.25, 10000)


PicPath = ''
PicFiles = [ f for f in listdir(PicPath) if path.isfile(path.join(PicPath,f)) and f.strip().split('.')[-1]=='png']#
countflag = 0
for Pic in PicFiles:
    tmp = Image.open(path.join(PicPath, Pic))
    print(countflag)
    line3 = np.array(tmp)
    line3 = line3.flatten()
    if countflag == 0:
        AllPx = line6
    else:
        AllPx = np.vstack((AllPx, line6))

    countflag = countflag + 1
    if countflag >= NumBatch:
        break

xx = AllPx
# Get 100 f(x) values
#p = norm()
#x = np.linspace(norm.ppf(0.01, loc=3, scale=2), norm.ppf(0.99, loc=3, scale=2), 100)
#p1 = norm.pdf(x, loc=3, scale=2)
#xx = np.reshape(x, (100, 1))

schedule = ais.get_schedule(100, rad=4)
print(schedule)
p2 = np.exp(model.ais(xx, schedule))
print(p2)

lld = model.ais(xx, schedule)
print('lld_mean:')
print(np.mean(lld))
plt.plot(x, p1)
plt.plot(x, p2)
plt.show()
