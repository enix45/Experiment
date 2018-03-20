import numpy as np
import tensorflow as tf
import gc
import math

from scipy.sparse import load_npz

class quick_recommend:
	def __init__(self, item_feat, K):
		self.input_dim = item_feat.shape[1]
		#self.item_feat = item_feat
		self.i_feat = tf.Variable(item_feat.T, dtype = tf.float32)
		self.k = K
		self._set_graph()

	def _set_graph(self):
		# Input
		self.u_input = tf.placeholder(tf.float32, shape = [None, self.input_dim])

		# Computation
		self.score = tf.matmul(self.u_input, self.i_feat)
		self.ans = tf.nn.top_k(self.score, k = self.k)[1]

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def predict_all(self, user_feat, batch_size = 500):
		nb_data = user_feat.shape[0]
		nb_batch = math.ceil(nb_data / batch_size)
		pred = list()
		for i in range(nb_batch):
			ans = self.sess.run([self.ans], feed_dict = {self.u_input: user_feat[i*batch_size:min((i+1) * batch_size, nb_data)]})
			pred.extend(list(ans[0]))
		return pred

	def reverse_dict(self, X):
		rev = {X[k]: k for k in X.keys()}
		return rev

	def load_dict(self, item_dict):
		self.item_dict = self.reverse_dict(item_dict)

	def predict_id(self, user_feat, batch_size = 500):
		ans = list()
		nb_data = user_feat.shape[0]
		pred = self.predict_all(user_feat)
		for i in range(nb_data):
			temp = list()
			for j in range(self.k):
				temp.append(self.item_dict[pred[i][j]])
			ans.append(temp)
		return ans


