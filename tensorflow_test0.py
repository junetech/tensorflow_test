'''
Short program from:
https://www.tensorflow.org/install/install_linux#CommonInstallationProblems
Written on May 31st 2018
'''
import os
import tensorflow as tf

if os.name == 'posix':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

