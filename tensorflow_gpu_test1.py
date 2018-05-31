'''
Test for GPU calculation on Fedora 28 with CUDA 9.2
Use by command: `python matmul.py gpu 1500` or `python matmul.py cpu 900`

Code mainly from: https://learningtensorflow.com/lesson10/
Written on May 31th, 2018 by JuneTech
'''

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import csv

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()
nowDatetime = startTime.strftime('%Y%m%d-%H%M%S')
filename = nowDatetime+'_'+"result.csv"
with open(filename, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(nowDatetime)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(sum_operation)
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writer.writerow(result)

with open(filename, 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow("Shape:", shape, "Device:", device_name)
    writer.writerow("Time taken:", datetime.now() - startTime)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)
