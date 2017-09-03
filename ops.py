import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
  rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
  tf.RegisterGradient(rnd_name)(grad)
  g = tf.get_default_graph()
  with g.gradient_override_map({"PyFunc": rnd_name}):
    return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def mysign(x, name=None):
  with ops.name_scope(name, "Mysign", [x]) as name:
    sign_x = py_func(np.sign,[x],[tf.float32],name=name,grad=_MySignGrad)
    return sign_x[0]

def _MySignGrad(op, grad):
  x = op.inputs[0]
  result = tf.add(tf.scalar_mul(0.5,tf.negative(tf.sign(tf.subtract(x,tf.ones(x.get_shape()))))),tf.ones(x.get_shape()))
  #print(result)
  return tf.multiply(result,grad)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d_d1(input_, output_dim, 
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="conv2d_d1"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))

    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
def conv2d_d2(input_, output_dim, 
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="conv2d_d2"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))

    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
def bconv2d_d1(input_, output_dim, 
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="conv2d_d2"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    n=k_h*k_w*w.get_shape().as_list()[2]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=0)/n

    conv = alpha*tf.nn.conv2d(input_, mysign(w), strides=[1, d_h, d_w, 1], padding='SAME')

    #XNOR supposed there is no biases
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def bconv2d_d2(input_, output_dim, 
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="conv2d_d2"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    n=k_h*k_w*w.get_shape().as_list()[2]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=0)/n

    conv = alpha*tf.nn.conv2d(input_, mysign(w), strides=[1, d_h, d_w, 1], padding='SAME')

    #XNOR supposed there is no biases
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv



def deconv2d_d1(input_, output_shape,
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="deconv2d_d1", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
def deconv2d_d2(input_, output_shape,
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d_d2", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
def bdeconv2d_d1(input_, output_shape,
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="deconv2d_d1", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    n=k_h*k_w*w.get_shape().as_list()[2]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=1)/n

    deconv = tf.nn.conv2d_transpose(input_, mysign(w), output_shape=output_shape,strides=[1, d_h, d_w, 1])
    deconv = tf.multiply(deconv,alpha)
    
    #XNOR supposed there is no biases
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
def bdeconv2d_d2(input_, output_shape,
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d_d2", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    n=k_h*k_w*w.get_shape().as_list()[2]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=1)/n

    deconv = tf.nn.conv2d_transpose(input_, mysign(w), output_shape=output_shape,strides=[1, d_h, d_w, 1])
    deconv = tf.multiply(deconv,alpha)
    
    #XNOR supposed there is no biases
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
def binconv2d_d1(input_, output_dim, 
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="binconv2d_d1"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    c=input_.get_shape()[-1].value #the depth of the input tensor
    A=tf.reduce_sum(tf.abs(input_),3,keep_dims=True)/c
    k=tf.ones([k_h,k_w,1,1])/(k_h*k_w)
    K=tf.nn.conv2d(A,k,strides=[1,d_h,d_w,1],padding='SAME')

    
    #there should be a better way to do this,calculate the norm of wn
    #wn is the L1 norm of w
    #but the last dimension of w is batch size and axis=[0,1] doesn't work
    n=k_h*k_w*w.get_shape().as_list()[2]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=0)/n

    signI=mysign(input_)
    signW=mysign(w)
    binIconvW=tf.nn.conv2d(signI, signW, strides=[1, d_h, d_w, 1], padding='SAME')
    conv=tf.multiply(binIconvW,K)
    conv=tf.multiply(conv,alpha)

    
    #XNOR supposed there is no biases
    #biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
def binconv2d_d2(input_, output_dim, 
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="binconv2d_d2"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    c=input_.get_shape()[-1].value #the depth of the input tensor
    A=tf.reduce_sum(tf.abs(input_),3,keep_dims=True)/c
    k=tf.ones([k_h,k_w,1,1])/(k_h*k_w)
    K=tf.nn.conv2d(A,k,strides=[1,d_h,d_w,1],padding='SAME')

    
    #there should be a better way to do this,calculate the norm of wn
    #wn is the L1 norm of w
    #but the last dimension of w is batch size and axis=[0,1] doesn't work
    n=k_h*k_w*w.get_shape().as_list()[2]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=0)/n

    signI=mysign(input_)
    signW=mysign(w)
    binIconvW=tf.nn.conv2d(signI, signW, strides=[1, d_h, d_w, 1], padding='SAME')
    conv=tf.multiply(binIconvW,K)
    conv=tf.multiply(conv,alpha)

    
    #XNOR supposed there is no biases
    #biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
def bindeconv2d_d1(input_, output_shape,
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="bindeconv2d_d1", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    c=input_.get_shape()[-1].value
    A=tf.reduce_sum(tf.abs(input_),3,keep_dims=True)/c
    k=tf.ones([k_h,k_w,1,1])/(k_h*k_w)

    K=tf.nn.conv2d_transpose(A,k,output_shape=[output_shape[0],output_shape[1],output_shape[2],1],strides=[1,d_h,d_w,1])

    n=k_h*k_w*w.get_shape().as_list()[3]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=1)/n

    signI=mysign(input_)
    signW=mysign(w)
    binIdeconvW=tf.nn.conv2d_transpose(signI, signW,output_shape=output_shape, strides=[1, d_h, d_w, 1])

    deconv=tf.multiply(binIdeconvW,K)
    deconv=tf.multiply(deconv,alpha)
    #XNOR supposed there is no biases
    #biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w
    else:
      return deconv
def bindeconv2d_d2(input_, output_shape,
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="bindeconv2d_d2", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    c=input_.get_shape()[-1].value
    A=tf.reduce_sum(tf.abs(input_),3,keep_dims=True)/c
    k=tf.ones([k_h,k_w,1,1])/(k_h*k_w)

    K=tf.nn.conv2d_transpose(A,k,output_shape=[output_shape[0],output_shape[1],output_shape[2],1],strides=[1,d_h,d_w,1])

    n=k_h*k_w*w.get_shape().as_list()[3]
    alpha=tf.norm(tf.norm(tf.norm(w,ord=1,axis=0),ord=1,axis=0),ord=1,axis=1)/n

    signI=mysign(input_)
    signW=mysign(w)
    binIdeconvW=tf.nn.conv2d_transpose(signI, signW,output_shape=output_shape, strides=[1, d_h, d_w, 1])

    deconv=tf.multiply(binIdeconvW,K)
    deconv=tf.multiply(deconv,alpha)
    #XNOR supposed there is no biases
    #biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w
    else:
      return deconv
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
