import tensorflow as tf
import numpy as np 
from Blackjack_ import BlackjackEnv
from collections import defaultdict
import plotting
import time

class Actor(object):
	def __init__(self, learning_rate=0.003):
		self.learning_rate = learning_rate
		self._build_model()

	def _build_model(self):
		n_hidden_1 = 8
		n_hidden_2 = 8
		n_input = 3
		n_class = 2

		self.x_pl = tf.placeholder(tf.float32, [None, n_input])
		self.y_pl = tf.placeholder(tf.int32, [None, n_class])
		self.td_error = tf.placeholder(tf.float32, [None])

		w1 = self.weight_variable([n_input, n_hidden_1])
		b1 = self.bias_variable([n_hidden_1])
		w2 = self.weight_variable([n_hidden_1, n_hidden_2])
		b2 = self.bias_variable([n_hidden_2])
		w3 = self.weight_variable([n_hidden_2, n_class])
		b3 = self.bias_variable([n_class])

		layer1 = tf.add(tf.matmul(self.x_pl, w1), b1)
		layer1 = tf.nn.relu(layer1)
		layer2 = tf.add(tf.matmul(layer1, w2), b2)
		layer2 = tf.nn.relu(layer2)
		self.softmax_input = tf.add(tf.matmul(layer2, w3), b3)
		self.all_act_prob = tf.nn.softmax(self.softmax_input, name='act_prob')
		self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
				logits=self.softmax_input, labels=self.y_pl)
		self.exp = tf.reduce_mean(self.neg_log_prob * self.td_error)
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.exp)

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)


	def choose_action(self, sess, s):
		prob_weights = sess.run(self.all_act_prob, feed_dict={self.x_pl:s})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

	def update(self, sess, s, a, td_error):
		one_hot_action = np.zeros(2)
		one_hot_action[a] = 1
		#reshape a
		aa = one_hot_action[np.newaxis, :]
		# train on episode
		sess.run(self.train_op, feed_dict={
			 self.x_pl: s,
			 self.y_pl: aa,
			 self.td_error: td_error,
		})

class Estimator(object):
	def __init__(self, learning_rate=0.003):
		self.learning_rate = learning_rate
		self._build_model()

	def _build_model(self):
		n_hidden_1 = 8
		n_hidden_2 = 8
		n_input = 3
		n_class = 1

		self.x_pl = tf.placeholder(tf.float32, [None, n_input])
		self.y_pl = tf.placeholder(tf.float32, [None, n_class])

		w1 = self.weight_variable([n_input, n_hidden_1])
		b1 = self.bias_variable([n_hidden_1])
		w2 = self.weight_variable([n_hidden_1, n_hidden_2])
		b2 = self.bias_variable([n_hidden_2])
		w3 = self.weight_variable([n_hidden_2, n_class])
		b3 = self.bias_variable([n_class])

		layer1 = tf.add(tf.matmul(self.x_pl, w1), b1)
		layer1 = tf.nn.relu(layer1)
		layer2 = tf.add(tf.matmul(layer1, w2), b2)
		layer2 = tf.nn.relu(layer2)
		self.predictions = tf.add(tf.matmul(layer2, w3), b3)
		self.loss = tf.square(self.y_pl - self.predictions)
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	def predict(self, sess, s):
		return sess.run(self.predictions, feed_dict={self.x_pl:s})

	def update(self, sess, s, y):
		_ = sess.run(self.train_op, feed_dict={self.x_pl:s, self.y_pl:y})

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)	

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
	state_array = np.array(state_nor).reshape(1,3)
	return state_array

def ac_test4debug(sess, env, actor, estimator, GAMMA=0.9, episode_num=100):
	def print_observation(observation):
		score, dealer_score, usable_ace = observation
		print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
			score, usable_ace, dealer_score))

	def strategy(observation):
		score, dealer_score, usable_ace = observation
		return 0 if score >= 20 else 1

	for i_episode in range(episode_num):
		print('=====episode:{}======'.format(i_episode))
		observation = env.reset()
		for t in range(100):
			print(observation)
			s = state_process(observation)
			v = estimator.predict(sess, s)
			a = actor.choose_action(sess, s)
			print("Taking action: {}".format(["Stick", "Hit"][a]))
			observation, reward, done, _ = env.step(a)

			if done:
				target = reward
				td_error = np.squeeze(target - v, axis=0)
				actor.update(sess, s, a, td_error)
				estimator.update(sess, s, np.reshape(target, (1,1)))
				break

			else:
				s_ = state_process(observation)
				v_ = estimator.predict(sess, s_)
				target = reward + GAMMA * v_
				td_error = np.squeeze(target - v, axis=0)
				actor.update(sess, s, a, td_error)
				estimator.update(sess, s, np.reshape(target, (1,1)))

	V = defaultdict(float)
	all_state = []
	for i in range(11, 22):
		for j in range(1, 11):
			for k in [True, False]:
				all_state.append((i, j, k))
	for state in all_state:
		nor_state = state_process(state)
		V[state] = np.squeeze(estimator.predict(sess, nor_state))
	return V

def main():
	env = BlackjackEnv()
	actor = Actor()
	estimator = Estimator()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		V = ac_test4debug(sess, env, actor, estimator, episode_num=10000)
	plotting.plot_value_function(V, title='Optimal Value Function')

if __name__ == '__main__':
	main()


























