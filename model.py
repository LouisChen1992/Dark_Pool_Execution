import tensorflow as tf

class Parametric_Model:
	def __init__(self, model, V_max, input_dim=2, num_layer=0, hidden_size=4):
		self._model = model
		self._input_dim = input_dim
		self._num_layer = num_layer
		self._hidden_size = hidden_size
		self._V_max = V_max

		self._build_placeholder()
		self._build_forward_pass_graph()
		self._build_train_op()

	def _build_placeholder(self):
		self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(self._V_max, self._input_dim))
		self.s_placeholder = tf.placeholder(dtype=tf.int32, shape=(self._V_max))
		self.n_placeholder = tf.placeholder(dtype=tf.int32, shape=(self._V_max))
		self.dp_placeholder = tf.placeholder(dtype=tf.float32, shape=())

	def _build_forward_pass_graph(self):
		with tf.variable_scope(self._model):
			self._W = dict()
			h_l = self.input_placeholder
			dim_l = self._input_dim
			for i in range(self._num_layer):
				self._W[i] = tf.get_variable(name='W%d'%i, shape=(dim_l, self._hidden_size))
				h_l = tf.nn.dropout(tf.nn.relu(tf.matmul(h_l, self._W[i])), keep_prob=self.dp_placeholder)
				dim_l = self._hidden_size
			self._W[self._num_layer] = tf.get_variable(name='W%d'%self._num_layer, shape=(dim_l, 1))
			logits = tf.matmul(h_l, self._W[self._num_layer])
			self.h = tf.reshape(1 / (1 + tf.exp(-logits)), [-1])

			logits = tf.concat([tf.zeros([self._V_max, 1]), -logits], axis=1)
			shape_n = tf.concat([tf.expand_dims(self.n_placeholder, 1), tf.ones([self._V_max, 1], dtype=tf.int32)], axis=1)
			shape_s = tf.concat([tf.expand_dims(self.s_placeholder, 1), tf.ones([self._V_max, 1], dtype=tf.int32)], axis=1)
			shape_n_s = tf.concat([tf.expand_dims(self.n_placeholder-self.s_placeholder, 1), tf.ones([self._V_max, 1], dtype=tf.int32)], axis=1)

			logits = tf.concat([tf.tile(tf.expand_dims(logits[i,:], 0), shape_n[i]) for i in range(self._V_max)], axis=0)
			labels = tf.concat([tf.concat([tf.tile(tf.zeros((1,1), dtype=tf.int32), shape_s[i]), tf.tile(tf.ones((1,1), dtype=tf.int32), shape_n_s[i])], axis=0) for i in range(self._V_max)], axis=0)
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labels, [-1]), logits=logits))

	def _build_train_op(self):
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss)
