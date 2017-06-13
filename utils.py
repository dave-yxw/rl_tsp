import os
import json
import logging
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.slim as slim

def prepare_dirs_and_logger(config):
  formatter = logging.Formatter(
      "%(asctime)s:%(levelname)s::%(message)s")
  logger = logging.getLogger('tensorflow')

  for hdlr in logger.handlers:
    logger.removeHandler(hdlr)

  handler = logging.StreamHandler()
  handler.setFormatter(formatter)

  logger.addHandler(handler)
  logger.setLevel(tf.logging.INFO)

  # model_name = 'tsp_2017-04-17_10-36-33'
  if config.load_path:
    if config.load_path.startswith(config.task):
      config.model_name = config.load_path
    else:
      config.model_name = "{}_{}".format(config.task, config.load_path)
  else:
    config.model_name = "{}_{}".format(config.task, get_time())

  # model_dir = "logs/" + model_name
  config.model_dir = os.path.join(config.log_dir, config.model_name)

  # mkdir if not exists
  for path in [config.log_dir, config.data_dir, config.model_dir]:
    if not os.path.exists(path):
      os.makedirs(path)

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_config(model_dir, config):
  """ save model config to model_dir/paras.json"""
  param_path = os.path.join(model_dir, "params.json")

  tf.logging.info("MODEL dir: %s" % model_dir)
  tf.logging.info("PARAM path: %s" % param_path)

  with open(param_path, 'w') as fp:
    json.dump(config.__dict__, fp,  indent=4, sort_keys=True)

def index_matrix_to_pairs_fn(batch_size, seq_length):
  replicated_first_indices = tf.range(batch_size) # range(128)
  # replicated_first_indices =
  #    [[  0,  0,  0,...],
  #     [  1,  1,  1,...],
  #     ......
  #     [127,127,127,...]]
  replicated_first_indices2 = tf.tile(
    tf.expand_dims(replicated_first_indices, dim=1), # [128,1]
    [1, seq_length])
  def index_matrix_to_pairs(index_matrix):
    """
    :param index_matrix: [batch_size, data_len] or [batch_size]
    :return: [batch_size, data_len, 2] or [batch_size, 2]
    ie:
      a: [128, 10] -> c[i,j,:] = [i,a[i,j]], shape(c) = [128,10,2]
      a: [128] -> c[i,:] = [i,a[i]], shape(c) = [128,2]
    """
    rank = len(index_matrix.get_shape())
    if rank == 1:
      return tf.stack([replicated_first_indices, index_matrix], axis=rank)
    elif rank == 2:
      return tf.stack([replicated_first_indices2, index_matrix], axis=rank)
    else:
      raise NotImplementedError("index_matrix rank should be 1 or 2, but %d found" % rank)

  return index_matrix_to_pairs

from config import get_config
config, unparsed = get_config()
index_matrix_to_pairs = index_matrix_to_pairs_fn(config.batch_size, config.data_length)
