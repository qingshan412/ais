import tensorflow as tf
import numpy as np

from parzen import tf_log_mean_exp


class ParsenDensityEstimator(object):
    def logpdf(self, x, mu, sigma):
        """
        Calculate the logpdf.
        :param x: [batch_size, output_dim]
        :param mu: Shape [num_samples, batch_size, output_dim]
        :param sigma: variance
        :return:
        """
        # J.L. 
        #d = (tf.expand_dims(x, 0) - mu) / sigma
        #e = -0.5 * tf.reduce_sum(tf.multiply(d, d), axis=2)
        #z = tf.to_float(tf.shape(mu)[2]) * tf.log(np.float32(sigma * np.sqrt(np.pi * 2.0)))
        d = (tf.expand_dims(x, 0) - mu) 
        e = -0.5 * tf.reduce_sum(tf.multiply(d, d) / sigma, axis=2)
        z = tf.to_float(tf.shape(mu)[2]) * tf.log(np.float32(np.sqrt(sigma * np.pi * 2.0)))
        return e - z

#-T.sum(T.square(generated-T.addbroadcast(self.obs,0)),[-1,-2]) / (2*self.sigma)
#- k/2.*np.log(2 * np.pi)
#- k/2.*np.log(self.sigma)
