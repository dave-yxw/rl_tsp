# encoding: UTF-8

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *
from utils import index_matrix_to_pairs

class Model(object):
  def __init__(self, config,
               summary_writer, reuse=False):
    self.task = config.task
    self.debug = config.debug
    self.config = config
    self.summary_writer = summary_writer

    self.input_dim = config.input_dim  # 2
    self.hidden_dim = config.hidden_dim  # 256
    self.num_layers = config.num_layers # 1

    self.data_length = config.data_length # 10
    self.num_glimpse = config.num_glimpse # 1

    self.init_min_val = config.init_min_val # -0.08
    self.init_max_val = config.init_max_val #  0.08
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

    self.lr_start = config.lr_start
    self.lr_decay_step = config.lr_decay_step
    self.lr_decay_rate = config.lr_decay_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}
    batch_size = self.config.batch_size

    with tf.device("gpu:0"):
      ##############
      # inputs
      ##############
      self.enc_inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, config.data_length, config.input_dim], name="enc_inputs")
      self.decode_target_ids = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.data_length], name="decoder_target_ids") # [batch_size, data_len]
      self.td = tf.placeholder(dtype=tf.float32, shape=[batch_size], name="td")  # V - reward = len - Y
      self.base_length = tf.placeholder(dtype=tf.float32, shape=[batch_size], name="base_length")
      self._build_model()
      self._build_length()
      self._build_optim()

      self.sample_summary = tf.summary.merge([
          tf.summary.scalar("sample/length", tf.reduce_mean(self.sampled_length)),
      ])
      self.train_summary = tf.summary.merge([
        self.train_summary,
        tf.summary.scalar("train/td", tf.reduce_mean(self.td)),  # 算是替代reward的监控了，big is better
      ])
      self.sample_train_summary = tf.summary.merge([
          self.sample_train_summary,
          self.sample_summary,
          tf.summary.scalar("train/td", tf.reduce_mean(self.sampled_td)), # Y - len
      ])
      self.test_summary = tf.summary.merge([
          tf.summary.scalar("test/diff", tf.reduce_mean(self.inference_diff)),
          tf.summary.scalar("test/length", tf.reduce_mean(self.inference_length))
      ])

  def _run(self, sess, fetch, feed_dict=None, do_summary=None, summary=None):
      """
      :param sess: 
      :param fetch: 需要执行的ops
      :param summary_writer: 
      :param summary: 需要执行的summary_ops 
      :return: 
      """
      fetch['step'] = self.global_step
      if do_summary is not None:
          fetch['summary'] = summary

      result = sess.run(fetch, feed_dict)
      if do_summary is not None:
          self.summary_writer.add_summary(result['summary'], result['step'])
          self.summary_writer.flush()
      return result

  def rl_sample(self, sess, batch_points, do_summary=None):  # sample a path/action
      result = self._run(sess, fetch={'x_path': self.sampled_path},
                         feed_dict={self.enc_inputs: batch_points},
                         do_summary=do_summary,
                         summary=self.sample_summary
                       )
      return result['x_path']

  def rl_train(self, sess, batch_points, target_path, td, do_summary=None):
      result = self._run(sess, fetch={'optim': self.policy_loss_optim},
                         feed_dict={
                           self.enc_inputs: batch_points,
                           self.decode_target_ids: target_path,
                           self.td: td
                       }, do_summary=do_summary, summary=self.train_summary)
      # print("policy_loss:", result['policy_loss'])

  def rl_sample_and_train(self, sess, batch_points, base_length, do_summary=None):
      result = self._run(sess, fetch={'optim': self.sampled_policy_loss_optim,
                                      'length': self.sampled_length},
                         feed_dict={self.enc_inputs: batch_points,
                                    self.base_length: base_length},
                         do_summary=do_summary, summary=self.sample_train_summary)
      # print("sampled_policy_loss:", result['sampled_policy_loss'])
      return result

  def sl_train(self, sess, batch_points, target_path, do_summary=None):
      result = self._run(sess, fetch={'optim': self.mle_loss_optim,},
                         feed_dict={
                             self.enc_inputs: batch_points,
                             self.decode_target_ids: target_path,
                         }, do_summary=do_summary, summary=self.mle_train_summary)

  def test(self, sess, batch_points, base_length, do_summary=None):
      result = self._run(sess, fetch={'x_path': self.inference_path,
                                      'length': self.inference_length},
                         feed_dict={self.enc_inputs: batch_points,
                                    self.base_length: base_length},
                         do_summary=do_summary, summary=self.test_summary)
      return result['x_path'], result['length']


  def _build_model(self):
    tf.logging.info("Create a model..")
    self.global_step = tf.Variable(0, trainable=False)

    batch_size = self.config.batch_size
    with tf.variable_scope("encoder"):
      # 对每个2D点套用2*256的embed,
      # 等价于在10*2(channel)上套用(filter=1, input_channel=2, output_channel=256, stride=1)的conv1D
      input_embed = tf.get_variable(
          "input_embed", [1, self.input_dim, self.hidden_dim],
          initializer=self.initializer)
      # embeded_enc_inputs: [batch_size, data_len, hidden_dim=256]
      embeded_enc_inputs = tf.nn.conv1d(
        self.enc_inputs, input_embed, 1, "VALID")
      enc_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [enc_cell] * self.num_layers
        enc_cell = MultiRNNCell(cells)
      enc_init_state = trainable_initial_state(
          batch_size, enc_cell.state_size)

      # encoder_outputs : [None, seq_length, hidden_dim]
      enc_outputs, enc_final_states = tf.contrib.rnn.static_rnn(
          enc_cell, tf.unstack(embeded_enc_inputs, axis=1),
          sequence_length=[self.data_length] * batch_size,
          initial_state=enc_init_state)
      enc_outputs = tf.stack(enc_outputs, axis=1)
          # tf.nn.dynamic_rnn(
          # enc_cell, embeded_enc_inputs,
          # [self.data_length] * batch_size, enc_init_state)

    with tf.variable_scope("decoder") as scope:
      dec_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [dec_cell] * self.num_layers
        dec_cell = MultiRNNCell(cells)

      sampled_logits, sampled_path, _ = ptn_rnn_decoder(
          dec_cell, None,
          embeded_enc_inputs, enc_outputs, enc_final_states,
          self.data_length, self.hidden_dim,
          self.num_glimpse, batch_size,
          initializer=self.initializer, mode="SAMPLE")
      # logits: [batch_size, step, data_len]
      self.sampled_logits = tf.identity(sampled_logits, name="sampled_logits")
      # sample_path: [batch_size, step]
      self.sampled_path = tf.identity(sampled_path, name="sampled_path")

      decoder_logits, _ = ptn_rnn_decoder(
          dec_cell, self.decode_target_ids,
          embeded_enc_inputs, enc_outputs, enc_final_states,
          self.data_length, self.hidden_dim,
          self.num_glimpse, batch_size,
          initializer=self.initializer, mode="TRAIN", reuse=True)
      self.dec_logits = tf.identity(decoder_logits, name="dec_logits")

      _, infered_path, _ = ptn_rnn_decoder(
          dec_cell, None,
          embeded_enc_inputs, enc_outputs, enc_final_states,
          self.data_length, self.hidden_dim,
          self.num_glimpse, batch_size,
          initializer=self.initializer, mode="BEAMSEARCH",
          reuse=True, beam_size=self.config.beam_size)
      self.inference_path = tf.identity(infered_path, name="infered_path")

  def _build_length(self):
      """
      根据sampled path直接计算reward
      :return: 
      """
      def shift_tensor(t, axis, step=1):
          unstacked_tl = tf.unstack(t, axis=axis)
          shifted_tl = unstacked_tl[step:]
          shifted_tl.extend(unstacked_tl[0:step])
          return tf.stack(shifted_tl, axis=axis)

      def compute_length(x_path):
          # path_points: [batch_size, data_len, data_dim]
          path_points = tf.gather_nd(self.enc_inputs,
                       index_matrix_to_pairs(x_path))
          shifted_path_points = shift_tensor(path_points, axis=1)
          length = tf.reduce_sum(
                tf.norm(path_points - shifted_path_points, axis=2),
                  axis=1)
          return length

      self.sampled_length = tf.stop_gradient(compute_length(self.sampled_path))
      self.sampled_td = tf.stop_gradient(self.base_length - self.sampled_length)
      self.inference_length = compute_length(self.inference_path)
      self.inference_diff = self.inference_length - self.base_length

  def _build_value(self):
      """
      根据tsp problem计算value net
      :return: 
      """
      pass

  def _build_optim(self):
    # sampled_policy_loss: cross_entropy(sampled_logits, sampled_path) * (reward_net(sampled_path) - base_value)
    # todo: base_value = value_net(sampled_path)
    # policy_loss: cross_entropy(dec_logits, decode_target_ids) * input_td, input_td = input_reward - base_value
    # cross_entropy: cross_entropy(dec_logits, decode_target_ids)
    # todo: value_loss: ||input_reward - value_net(input_path) ||^2


    # 训练过程中，cross_entropy = -log(pi) > 0, 会趋向于0。当策略pi变成确定性策略时cross_entropy为0
    # 最大似然训练目标: min corss_entropy
    # policy_loss = entropy_loss * td, td = Y-len <= 0.
    # rl训练目标: min policy_loss
    # cross_entropy(+) -> 0, td(-) -> 0, 所以policy loss(-) -> 0
    sampled_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sampled_path, logits=self.sampled_logits)
    self.sampled_policy_loss = tf.reduce_mean(
        tf.reduce_sum(sampled_cross_entropy, axis=1) * self.sampled_td,
        name="sampled_policy_loss"
    )
    if self.config.entropy_reg > 0:
        # 加entropy loss提升稳定性
        # pi: [batch_size, seq_len, data_len]
        pi = tf.nn.softmax(self.sampled_logits, dim=-1)
        # entropy = -\sum_i pi log(pi) >= 0, 当p时one hot时，entropy达到最小值0
        # 正则化: max entropy
        sampled_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.sampled_logits, labels=pi)
        sampled_entropy_reg = tf.reduce_mean(
            tf.reduce_sum(sampled_entropy, axis=1), name="sampled_entropy")
        self.sampled_policy_loss = self.sampled_policy_loss - self.config.entropy_reg * sampled_entropy_reg

    # entropy_loss: [batch_size]
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.decode_target_ids, logits=self.dec_logits)
    self.policy_loss = tf.reduce_mean(
        tf.reduce_sum(cross_entropy, axis=1) * self.td, name="policy_loss")
    self.mle_loss = tf.reduce_sum(cross_entropy, name="mle_loss")
    if self.config.entropy_reg > 0:
        pi = tf.nn.softmax(self.dec_logits, dim=-1)
        entropy = \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.dec_logits, labels=pi)
        entropy_reg = tf.reduce_mean(tf.reduce_sum(entropy, axis=1))
        self.policy_loss = self.policy_loss - self.config.entropy_reg * entropy_reg
        self.mle_loss = self.mle_loss - self.config.entropy_reg * entropy_reg

    self.lr = tf.train.exponential_decay(
        self.lr_start, self.global_step, self.lr_decay_step,
        self.lr_decay_rate, staircase=True, name="learning_rate")

    optimizer = tf.train.AdamOptimizer(self.lr)

    def build_loss_opt(loss, name="0"):
      grads_and_vars = optimizer.compute_gradients(loss)

      if self.max_grad_norm != None:
        for idx, (grad, var) in enumerate(grads_and_vars):
            if grad is not None:
                grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

      var_summary = tf.summary.merge( [tf.summary.scalar(name+"_varnorm/"+var.name, tf.norm(var)) for grad, var in grads_and_vars] )
      grad_summary = tf.summary.merge( [tf.summary.scalar(name+"_gradnorm/"+var.name, tf.norm(grad)) for grad, var in grads_and_vars] )
      return optim, tf.summary.merge([
          var_summary, grad_summary,
          tf.summary.scalar("train/loss", tf.reduce_mean(loss)),
          tf.summary.scalar("train/lr", self.lr)])
    self.policy_loss_optim, self.train_summary = build_loss_opt(self.policy_loss, name='policy_loss')
    self.sampled_policy_loss_optim, self.sample_train_summary = build_loss_opt(self.sampled_policy_loss, name='sample_policy_loss')
    self.mle_loss_optim, self.mle_train_summary = build_loss_opt(self.mle_loss, name='mle_loss')
