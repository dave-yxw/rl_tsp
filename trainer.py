#-*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
from data_loader import TSPDataLoader

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step  # default 50
    self.max_step = config.max_step  # default 1e6
    self.num_log_samples = config.num_log_samples  # default 3
    self.checkpoint_secs = config.checkpoint_secs  # defaut 300

    self.data_loader = TSPDataLoader(config, rng=self.rng)

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    self.model = Model(
        config,
        summary_writer = self.summary_writer
    )
    self._build_session()
    show_all_variables()

  def _build_session(self):
    self.saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.global_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def rl_train(self):
    tf.logging.info("Training starts...")
    for k in trange(self.max_step, desc="train"):
      x, _, y = self.data_loader.get_train_batch(self.config.batch_size)

      do_summary = True if (k+1) % self.log_step == 0 else None
      # x_path: [batch_size, data_len]
      # x_path = self.model.sample(self.sess, batch_points=x)

      # 根据x和path生成reward = -len
      # from data_loader import get_path_distance
      # length = [get_path_distance(x[i,:,:], x_path[i,:])
      #   for i in range(self.config.batch_size)]
      # self.model.train(self.sess, batch_points=x, target_path=x_path, td=y - length,
      #                  do_summary=do_summary)
      result = self.model.rl_sample_and_train(self.sess,
                                     batch_points=x,
                                     base_length=y,
                                     do_summary=do_summary)
      # self.data_loader.update_last_batch_critic(result['length']*0.3 + y*0.7)

      # todo: better if data_loader单独提供一份测试数据集
      if (k % 100) == 0:
          x, _, y = self.data_loader.get_test_batch(self.config.batch_size)
          self.model.test(self.sess, batch_points=x, base_length=y, do_summary=True)

  def test(self):
    """用于训练完成以后单独test"""
    tf.logging.info("Testing starts...")
    loop_num = int(self.data_loader.data_num['test']/self.config.batch_size)
    # 顺序遍历一遍数据集
    for _ in trange(loop_num, desc="test"):
        x, _, y = self.data_loader.get_test_batch(self.config.batch_size)
        x_path, x_length = self.model.test(self.sess,
                        batch_points=x,
                        base_length=y,
                        do_summary=True)
        # print x_length - y