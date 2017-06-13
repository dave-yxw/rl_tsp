# -*- coding: utf-8 -*-


# Most of the codes are from
# https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import threading
import re
from collections import deque
import zipfile
import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple
import concorde

import tensorflow as tf
from download import download_file_from_google_drive
import random

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

def get_path_distance(tsp_points, tsp_path):
  """
  :param tsp_points: len * dim 
  :param tsp_path: [len], a permutation of 0~(n-1)
  :return: tsp_distance
  """
  distance = 0
  path_len = tsp_points.shape[0]
  for i in range(path_len):
    distance += length(tsp_points[tsp_path[i]], tsp_points[tsp_path[(i+1)%path_len]])
  return distance

def generate_one_example(n_nodes, rng, thread_id=0):
  nodes = rng.rand(n_nodes, 2).astype(np.float32)
  # solutions = solve_tsp_dynamic(nodes)
  solutions = concorde.solve(nodes, tmp_dir="/tmp/tsp_%d" % thread_id)
  distance = get_path_distance(nodes, solutions)
  return [nodes, solutions, distance] # TSPSample(x=nodes, sol=solutions, y=distance)

class TSPDataLoader(object):
  """
    下载并解析数据，装入self.data[name]
    self.run_input_queue(): 开启喂数据线程，将数据从self.data喂入tf计算流
    self.x, self.y, self.seq_length, self.mask: 模型训练读数据用
  """
  def __init__(self, config, rng=None):
    self.config = config
    if rng is None:
      rng = np.random.RandomState()
    self.rng = rng

    self.task = config.task.lower()
    self.batch_size = config.batch_size  # default 128
    self.data_length = config.data_length # default 10

    self.is_train = config.is_train
    self.random_seed = config.random_seed

    self.data_num = {'test':config.test_num}
    self.test_queue = deque(maxlen=self.data_num['test'])
    if config.is_train:
      self.data_num['train'] = config.train_num
      self.train_queue = deque(maxlen=self.data_num['train'])

    self.data_dir = config.data_dir
    self.task_name = "{}_{}".format(  # tsp_(5,10)
        self.task, self.data_length)

    self.samples = {}

    # x: point on graph, sample_size * path_length * dim(TSP2D)
    # y: path by id, sample_size * path_length, ie. [1 5 8 6 4 10 2 3 9 7]
    for name in self.data_num:
      path = self.get_path(name)
      if os.path.exists(path):
        self.load_data(name)
      else:
        self._generate_and_save(name)
        
  def get_batch(self, q, batch_size, name):
    while len(q) < batch_size:
      random.shuffle(self.samples[name])
      q.extend(self.samples[name])

    self.last_batch = [q.pop() for _ in range(batch_size)]
    x,sol,opt = zip(*self.last_batch)
    return np.asarray(x), np.asarray(sol), np.asarray(opt)

  def get_test_batch(self, batch_size):
    return self.get_batch(self.test_queue, batch_size, 'test')

  def get_train_batch(self, batch_size):
    """ 
        生成一个batch数据
    :param batch_size: 
    :return: tuple(x,y), x:[batch_size, data_len, data_dim], y:[batch_size]
    """
    return self.get_batch(self.train_queue, batch_size, 'train')

  def save_data(self, name):
    path = self.get_path(name)
    tf.logging.info("save data to {} for [{}]".format(path, self.task))
    x, sol, y, _ = zip(*self.samples[name])
    x = np.asarray(x)
    y = np.asarray(y)
    sol = np.asarray(sol)
    np.savez(path, x=x, y=y, sol=sol)

  def load_data(self, name):
    path = self.get_path(name)
    tf.logging.info("load data from {} for [{}]".format(path, self.task))
    tmp = np.load(path)
    x = tmp['x']
    y = tmp['y']
    sol = tmp['sol']
    self.samples[name] = [[x[i], 0, y[i]]  # not neccessary to load best solution path now
      # TSPSample(x=x[i], sol=sol[i], y=y[i])
      for i in range(x.shape[0])]

  def _generate_and_save(self, name):
    """ generate data:
    test data:
     tsp_(5,10)_test=1000.npz, in numpy's binary format 
     self.data['test'] = [x,y,'test']
     x: point on graph, sample_size * max_length * dim(TSP2D), path_len可能小鱼max_length, 不足部分是默认值0
     y: path by id, sample_size * max_length
    """
    path = self.get_path(name)

    # generate data if not exists, else load
    if not os.path.exists(path):
      tf.logging.info("Creating {} for [{}]".format(path, self.task))

      samples = deque(maxlen=self.data_num[name])
      def gen_thread_fn(thread_id):
        pre_state = 0
        while len(samples) < self.data_num[name]:
          sample = generate_one_example(self.data_length, self.rng, thread_id=thread_id)
          sample.append(sample[-1])
          samples.append(sample)
          state = int(len(samples)*100 / self.data_num[name])
          if thread_id == 0 and state > pre_state:
            tf.logging.info("%d%%/%d data finished" % (state, self.data_num[name]))
            pre_state = state
        print "thread", thread_id, "exit"

      gen_threads = [threading.Thread(target=gen_thread_fn, args=[i])
                     for i in range(self.config.thread_num)]
      [t.start() for t in gen_threads]
      [t.join() for t in gen_threads]

      self.samples[name] = list(samples)
      self.save_data(name)

  def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

