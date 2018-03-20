import tensorflow as tf
import math
import numpy as np

from random import shuffle
from time import time
from scipy.sparse import csr_matrix, lil_matrix, load_npz

from tools import save_pickle, load_pickle

def lrelu(x, alpha = 0.2):
	return tf.maximum(x, alpha * x)

# In this version, the attention is derived from custno and applied on the output
class PCN:
	def __init__(self, u_input_dims, i_input_dims, emb_dim, user_dims, item_dims, att_dims, phase, params = None, lr = 1e-4, epsilon = 1e-6, att_bias = False):
		self.u_in_dim = u_input_dims
		self.i_in_dim = i_input_dims
		self.nb_u_in = len(u_input_dims)
		self.nb_i_in = len(i_input_dims)
		self.u_sidx = [sum(u_input_dims[:i]) for i in range(1, self.nb_u_in)]
		self.i_sidx = [sum(i_input_dims[:i]) for i in range(1, self.nb_i_in)]
		self.emb_dim = emb_dim
		self.u_dims = [(self.nb_u_in-1) * emb_dim]
		self.u_dims.extend(user_dims)
		self.i_dims = [self.nb_i_in * emb_dim]
		self.i_dims.extend(item_dims)
		self.lr = lr
		self.att_dims = [self.nb_u_in * emb_dim]
		self.att_dims.extend(att_dims)
		if att_bias:
			self.att_dims.append(2 * user_dims[-1])
		else:
			self.att_dims.append(user_dims[-1])
		self.phase = phase # phase 0 for pre-train, 1 from tune
		self.eps = epsilon
		self.att_bias = att_bias
		self.train_phase = True # For batch normalization
		self._set_weight(params)
		self._set_graph()
		if phase == 0 or phase == 2:
			self.train_rec = list()
		else:
			self.train_rec = [[], [], []]

	def _set_graph(self):
		# Input
		self.inputs = list()
		for i in range(self.nb_u_in):
			self.inputs.append(tf.placeholder(tf.float32, shape = [None, self.u_in_dim[i]]))
		for i in range(self.nb_i_in):
			self.inputs.append(tf.placeholder(tf.float32, shape = [None, self.i_in_dim[i]]))
		self.inputs.append(tf.placeholder(tf.float32))

		# Embedding
		self.cust = tf.matmul(self.inputs[0], self.u_emb[0])
		self.u_h = [tf.matmul(self.inputs[i], self.u_emb[i]) for i in range(1, self.nb_u_in)]
		self.i_h = [tf.matmul(self.inputs[i + self.nb_u_in], self.i_emb[i]) for i in range(self.nb_i_in)]
		self.u_h = tf.concat(self.u_h, 1)
		self.i_h = tf.concat(self.i_h, 1)

		# Attention Network 
		# Note that the attention is only for the user
		self.att = tf.concat([self.cust, self.u_h], 1)
		for i in range(len(self.att_dims) - 1):
			self.att = lrelu(tf.add(tf.matmul(self.att, self.att_param[i][0]), self.att_param[i][1]))
			if i != (len(self.att_dims) - 2):
				if self.train_phase:
					fc_mean, fc_var = tf.nn.moments(self.att, axes = [0])
					self.att_bn[i].apply([fc_mean, fc_var])
				self.att = tf.nn.batch_normalization(self.att, self.att_bn[i].average(fc_mean), self.att_bn[i].average(fc_var), self.att_param[i][2], self.att_param[i][3], self.eps)
		#self.att = tf.nn.sigmoid(self.att)
		if self.att_bias:
			self.att_b = self.att[:, self.u_dims[-1]:]
			self.att = self.att[:, :self.u_dims[-1]]
		self.att = 2 * tf.nn.tanh(self.att)
		if self.att_bias:
			self.att = tf.add(self.att, self.att_b)

		# Computation
		for i in range(len(self.u_dims) - 1):
			self.u_h = lrelu(tf.add(tf.matmul(self.u_h, self.u_params[i][0]), self.u_params[i][1]))
			if i != (len(self.u_dims) - 2):
				if self.train_phase:
					fc_mean, fc_var = tf.nn.moments(self.u_h, axes = [0])
					self.u_bn[i].apply([fc_mean, fc_var])
				self.u_h = tf.nn.batch_normalization(self.u_h, self.u_bn[i].average(fc_mean), self.u_bn[i].average(fc_var), self.u_params[i][2], self.u_params[i][3], self.eps)

		for i in range(len(self.i_dims) - 1):
			self.i_h = lrelu(tf.add(tf.matmul(self.i_h, self.i_params[i][0]), self.i_params[i][1]))
			if i != (len(self.u_dims) - 2):
				if self.train_phase:
					fc_mean, fc_var = tf.nn.moments(self.i_h, axes = [0])
					self.i_bn[i].apply([fc_mean, fc_var])
				self.i_h = tf.nn.batch_normalization(self.i_h, self.i_bn[i].average(fc_mean), self.i_bn[i].average(fc_var), self.i_params[i][2], self.i_params[i][3], self.eps)

		# The answer
		self.ans_0 = tf.nn.sigmoid(tf.matmul(self.u_h, tf.transpose(self.i_h)))
		self.pos_att = tf.multiply(self.u_h, self.att)
		self.ans_1 = tf.nn.sigmoid(tf.matmul(self.pos_att, tf.transpose(self.i_h)))

		# Loss 
		self.label = self.inputs[-1]
		self.b_w = tf.to_float(tf.size(self.label)) / tf.to_float(tf.reduce_sum(self.label)) 
		self.loss_0 = - self.b_w * tf.reduce_mean(tf.multiply(self.label, tf.log(self.ans_0 + self.eps))) - tf.reduce_mean(tf.multiply(1. - self.label, tf.log(1. + self.eps - self.ans_0)))
		self.loss_1 = - self.b_w * tf.reduce_mean(tf.multiply(self.label, tf.log(self.ans_1 + self.eps))) - tf.reduce_mean(tf.multiply(1. - self.label, tf.log(1. + self.eps - self.ans_1)))

		if self.phase == 0: # Pre-train
			self.loss = self.loss_0
		elif self.phase == 1: # Tune
			self.loss = self.loss_0 + self.loss_1
		elif self.phase == 2: # Only the attention part
			self.loss = self.loss_1
		else:
			raise NameError('No such phase')

		# Optimizer
		self.optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = self.optimizer.minimize(self.loss)

		# Session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _set_weight(self, params):
		self.u_params = list()
		self.i_params = list()
		self.u_bn = list()
		self.i_bn = list()
		self.att_param = list()
		self.att_bn = list()
		if params:
			self.u_emb = [tf.Variable(params[0][i]) for i in range(self.nb_u_in)]
			self.i_emb = [tf.Variable(params[1][i]) for i in range(self.nb_i_in)]
			#self.att_param = [[tf.Variable(params[4][i][j]) for j in range(2)] for i in range(2)]
		else:
			self.u_emb = [tf.Variable(tf.random_normal([self.u_in_dim[i], self.emb_dim], 0.0, 0.01)) for i in range(self.nb_u_in)]
			self.i_emb = [tf.Variable(tf.random_normal([self.i_in_dim[i], self.emb_dim], 0.0, 0.01)) for i in range(self.nb_i_in)]
			#self.att_param.append([tf.Variable(tf.random_normal([self.u_dims[0] + self.emb_dim, 256], 0.0, 0.01)), tf.Variable(tf.random_uniform([256], 0.0, 0.0))])
			#self.att_param.append([tf.Variable(tf.random_normal([256, self.u_dims[-1]], 0.0, 0.01)), tf.Variable(tf.random_uniform([self.u_dims[-1]], 0.0, 0.0))])
		for i in range(len(self.att_dims) - 1):
			if params:
				W = tf.Variable(params[4][i][0])
				b = tf.Variable(params[4][i][1])
				scale = tf.Variable(params[4][i][2])
				shift = tf.Variable(params[4][i][3])
			else:
				W = tf.Variable(tf.random_normal([self.att_dims[i], self.att_dims[i+1]], 0.0, 0.01))
				b = tf.Variable(tf.random_uniform([self.att_dims[i+1]], 0.0, 0.0))
				scale = tf.Variable(tf.ones([self.att_dims[i+1]]))
				shift = tf.Variable(tf.zeros([self.att_dims[i+1]]))
			self.att_param.append([W, b, scale, shift])
			self.att_bn.append(tf.train.ExponentialMovingAverage(decay = 0.5))
		for i in range(len(self.u_dims) - 1):
			if params:
				W = tf.Variable(params[2][i][0])
				b = tf.Variable(params[2][i][1])
				scale = tf.Variable(params[2][i][2])
				shift = tf.Variable(params[2][i][3])
			else:
				W = tf.Variable(tf.random_normal([self.u_dims[i], self.u_dims[i+1]], 0.0, 0.01))
				b = tf.Variable(tf.random_uniform([self.u_dims[i+1]], 0.0, 0.0))
				scale = tf.Variable(tf.ones([self.u_dims[i+1]]))
				shift = tf.Variable(tf.zeros([self.u_dims[i+1]]))
			self.u_params.append([W, b, scale, shift])
			self.u_bn.append(tf.train.ExponentialMovingAverage(decay = 0.5))
		for i in range(len(self.i_dims) - 1):
			if params:
				W = tf.Variable(params[3][i][0])
				b = tf.Variable(params[3][i][1])
				scale = tf.Variable(params[3][i][2])
				shift = tf.Variable(params[3][i][3])
			else:
				W = tf.Variable(tf.random_normal([self.i_dims[i], self.i_dims[i+1]], 0.0, 0.01))
				b = tf.Variable(tf.random_uniform([self.i_dims[i+1]], 0.0, 0.0))
				scale = tf.Variable(tf.ones([self.i_dims[i+1]]))
				shift = tf.Variable(tf.zeros([self.i_dims[i+1]]))
			self.i_params.append([W, b, scale, shift])
			self.i_bn.append(tf.train.ExponentialMovingAverage(decay = 0.5))

	def save_weight(self, file_name):
		params = self.sess.run([self.u_emb, self.i_emb, self.u_params, self.i_params, self.att_param])
		save_pickle(params, file_name)

	def load_positives(self, pos):
		self.pos_pair = pos

	def sample(self, user_idx):
		item_idx = set()
		for idx in user_idx:
			item_idx = item_idx.union(self.pos_pair[idx])
		return list(item_idx)

	def fit(self, u_feat, i_feat, label, nb_epoch, batch_size = 256):
		self.train_phase = True
		indices = list(self.pos_pair.keys())
		nb_data = len(indices)
		nb_batch = math.ceil(nb_data / batch_size)
		self.n_inputs = self.nb_u_in + self.nb_i_in + 1
		for epoch in range(nb_epoch):
			print('Epoch {}/{}:'.format(epoch + 1, nb_epoch))
			if self.phase == 0 or self.phase == 2:
				t_loss = list()
			else:
				t_loss = [[], [], []]
			shuffle(indices)
			b_time = time()
			for i in range(nb_batch):
				u_idx = indices[i * batch_size:min((i+1) * batch_size, nb_data)]
				i_idx = self.sample(u_idx)
				inputs = np.split(u_feat[u_idx].toarray(), self.u_sidx, axis = 1)
				inputs.extend(np.split(i_feat[i_idx].toarray(), self.i_sidx, axis = 1))
				t_gt = label[u_idx].tocsc()
				inputs.append(t_gt[:,i_idx].toarray())
				if self.phase == 0:
					_, c = self.sess.run(
							[self.train_op, self.loss], 
							feed_dict = {self.inputs[i]: inputs[i] for i in range(1, self.n_inputs)})
					t_loss.append(c)
					self.train_rec.append(c)
					print('%d/%d - Loss: %.5f, Avg. loss: %.5f       ' %(i+1, nb_batch, c, np.mean(t_loss)), end = '\r')
				elif self.phase == 2:
					_, c = self.sess.run(
							[self.train_op, self.loss], 
							feed_dict = {self.inputs[i]: inputs[i] for i in range(self.n_inputs)})
					t_loss.append(c)
					self.train_rec.append(c)
					print('%d/%d - Loss: %.5f, Avg. loss: %.5f       ' %(i+1, nb_batch, c, np.mean(t_loss)), end = '\r')
				else:
					_, a, b, c = self.sess.run(
							[self.train_op, self.loss_0, self.loss_1, self.loss], 
							feed_dict = {self.inputs[i]: inputs[i] for i in range(self.n_inputs)})
					t_loss[0].append(a)
					t_loss[1].append(b)
					t_loss[2].append(c)
					self.train_rec[0].append(a)
					self.train_rec[1].append(b)
					self.train_rec[2].append(c)
					print('%d/%d - Avg. loss 0: %.5f, Avg. loss 1: %.5f, Avg. total loss 2: %.5f       ' %(i+1, nb_batch, np.mean(t_loss[0]),  np.mean(t_loss[1]),  np.mean(t_loss[2])), end = '\r')
			if self.phase == 0 or self.phase == 2:
				print('%.1f sec - Loss: %.5f, Avg. loss: %.5f     ' %(time() - b_time, c, np.mean(t_loss)))
			else:
				print('%.1f sec - Avg. loss 0: %.5f, Avg. loss 1: %.5f, Avg. total loss: %.5f    ' %(time() - b_time, np.mean(t_loss[0]), np.mean(t_loss[1]), np.mean(t_loss[2])))

	def get_feature(self, in_feat, end, att = True, batch_size = 2000):
		self.train_phase = False
		nb_data = in_feat.shape[0]
		nb_batch = math.ceil(nb_data / batch_size)
		pred = list()
		for i in range(nb_batch):
			if end == 'user':
				i_f = np.split(in_feat[i * batch_size:min((i+1) * batch_size, nb_data)].toarray(), self.u_sidx, axis = 1)
				if att:
					ans = self.sess.run([self.pos_att], feed_dict = {self.inputs[i]: i_f[i] for i in range(self.nb_u_in)})
				else:
					ans = self.sess.run([self.u_h], feed_dict = {self.inputs[i]: i_f[i] for i in range(1, self.nb_u_in)})
			elif end == 'item':
				i_f = np.split(in_feat[i * batch_size:min((i+1) * batch_size, nb_data)].toarray(), self.i_sidx, axis = 1)
				ans = self.sess.run([self.i_h], feed_dict = {self.inputs[i + self.nb_u_in]: i_f[i] for i in range(self.nb_i_in)})
			else:
				raise NameError('No such bussiness end')
			pred.extend(list(ans[0]))
		return np.asarray(pred)



