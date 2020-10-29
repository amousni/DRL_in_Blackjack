import tensorflow as tf
import numpy as np 
from Blackjack_ import BlackjackEnv
from collections import defaultdict
import plotting
import time

#hyper parameters
MAX_EPISODES = 2000				#episodes for train
LR_A = 0.001					#learning rate of actor
LR_C = 0.002					#learning rate of critic
GAMMA = 0.99					#reward discount factor
TAU = 0.1						#update network parameter with TAU * w + (1-TAU) * w'
MEMORY_CAPACITY = 10			#max memory capacity
BATCH_SIZE = 32					#batch size for sampling from experience pool 


#DDPG
class DDPG(object):
	def __init__(self, a_dim, s_dim, a_bound,):
		self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
		#for memory update
		self.pointer = 0
		self.sess = tf.Session()

		self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
		self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
		self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
		self.R = tf.placeholder(tf.float32, [None, 1], 'r')
		self.done = tf.placeholder(tf.float32, [None, 1], 'done')

		with tf.variable_scope('Actor'):
			self.a = self._build_a(self.S, scope='eval', trainable=True)
			a_ = self._build_a(self.S_, scope='target', trainable=False)

		with tf.variable_scope('Critic'):
			self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
			q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

		self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
		self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
		self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
		self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

		self.soft_replace = [tf.assign(t, (1-TAU)*t + TAU*e) for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]


		'''
			need optimization
		'''
		q_target = self.R + GAMMA * q_ * (1.0 - self.done)
		td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
		self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

		a_loss = - tf.reduce_mean(self.q)
		self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

		self.sess.run(tf.global_variables_initializer())

	def choose_action(self, s):
		prob_weights = self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

	def predict(self, s, a):
		return self.sess.run(self.q, feed_dict={self.S:s[np.newaxis, :], self.a:a})

	def learn(self):
		self.sess.run(self.soft_replace)
		indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
		batch_data = self.memory[indices, :]
		bs = batch_data[:, :self.s_dim]
		ba = batch_data[:, self.s_dim:self.s_dim + self.a_dim]
		br = batch_data[:, -self.s_dim - 2: -self.s_dim - 1]
		bs_ = batch_data[:, -self.s_dim - 1: -1]
		bd = batch_data[:, -1]
		bd = np.reshape(bd, (BATCH_SIZE, 1))

		self.sess.run(self.atrain, feed_dict={self.S: bs})
		self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a:ba, self.R:br, self.S_:bs_, self.done:bd})

	def store_transition(self, s, a, r, s_, done):
		transition = np.hstack((s, a, [r], s_, [done]))
		index = self.pointer % MEMORY_CAPACITY
		self.memory[index, :] = transition
		self.pointer += 1

	def _build_a(self, s, scope, trainable):
		with tf.variable_scope(scope):
			net = tf.layers.dense(s, 8, activation=tf.nn.relu, name='l1', trainable=trainable)
			a = tf.layers.dense(net, self.a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)
			return tf.multiply(a, self.a_bound, name='scaled_a')

	def _build_c(self, s, a, scope, trainable):
		with tf.variable_scope(scope):
			w1_s = tf.get_variable('w1_s', [self.s_dim, 8], trainable=trainable)
			w1_a = tf.get_variable('w1_a', [self.a_dim, 8], trainable=trainable)
			b1 = tf.get_variable('b1', [1, 8], trainable=trainable)
			net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
			return tf.layers.dense(net, 1, trainable=trainable)

def state_process(state):
	#state process
	state_nor = []
	for i in range(len(state)):
		if i == 0:
			state_nor.append(state[i] / 32.0)
		elif i == 1:
			state_nor.append(state[i] / 22.0)
		else:
			if state[i] == True:
				state_nor.append(1.0)
			else:
				state_nor.append(0.0)
	state_array = np.array(state_nor).reshape(3)
	return state_array

def DDPG4debug():
	env = BlackjackEnv()
	s_dim = 3
	a_dim = 2
	a_bound = 1

	ddpg = DDPG(a_dim, s_dim, a_bound)

	t1 = time.time()

	for episode in range(MAX_EPISODES):
		print('=========episode: {}========'.format(episode))
		s = env.reset()
		s = state_process(s)
		while True:
			a = ddpg.choose_action(s)
			s_, r, done, _ = env.step(a)

			if done:
				done_normalized = 1.0
			else:
				done_normalized = 0.0

			s_ = state_process(s_)
			if a == 0:
				a_normalized = np.array([1, 0])
			elif a == 1:
				a_normalized = np.array([0, 1])

			print('---------')
			print(s)
			print(a_normalized)
			print(r)
			print(s_)
			#print(ddpg.predict(s, [[0,1]]))
			print('---------')
			ddpg.store_transition(s, a_normalized, r, s_, done_normalized)
			
			if ddpg.pointer > MEMORY_CAPACITY:
				ddpg.learn()

			if done:
				break
			else:
				s = s_

	V = defaultdict(float)
	all_state = []
	for i in range(11, 22):
		for j in range(1, 11):
			for k in [True, False]:
				all_state.append((i, j, k))
	for state in all_state:
		nor_state = state_process(state)
		V[state] = np.squeeze(max(ddpg.predict(nor_state, [[1,0]]), ddpg.predict(nor_state, [[0,1]])))
	return V

def main():
	V = DDPG4debug()
	#print(V)
	plotting.plot_value_function(V, title='Optimal Value Function')

if __name__ == '__main__':
	main()










