import tensorflow as tf
import numpy as np
from Blackjack_ import BlackjackEnv
from collections import defaultdict
import plotting

class Estimator(object):

    def __init__(self, learning_rate=0.003):
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):

        n_hidden_1 = 8
        n_hidden_2 = 8
        n_input = 3
        n_classes = 2

        self.x_pl = tf.placeholder(tf.float32, [None, n_input])
        self.y_pl = tf.placeholder(tf.float32, [None])
        self.action_pl = tf.placeholder(tf.int32, [None])

        batch_sz = tf.shape(self.x_pl)[0]

        self.w = {
            'w1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'w2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        self.b = {
            'b1':tf.Variable(tf.random_normal([n_hidden_1])),
            'b2':tf.Variable(tf.random_normal([n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_classes]))
        }

        l1 = tf.nn.relu(tf.add(tf.matmul(self.x_pl, self.w['w1']), self.b['b1']))
        l2 = tf.nn.relu(tf.add(tf.matmul(l1, self.w['w2']), self.b['b2']))
        self.predictions = tf.add(tf.matmul(l2, self.w['out']), self.b['out'])

        gather_indices = tf.range(batch_sz) * tf.shape(self.predictions)[1] + self.action_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        
        self.losses = tf.square(self.action_predictions - self.y_pl)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, s):
        return sess.run(self.predictions, feed_dict={self.x_pl:s})

    def update(self, sess, s, a, y):
        _ = sess.run(self.train_op, feed_dict={self.x_pl:s, self.y_pl:y, self.action_pl:a})

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #self.preictions:[[q(s, a_1), q(s, a_2)]]
        #best_action = np.argmax([q(s, a_1, q(s, a_2))])
        best_action = np.argmax(estimator.predict(sess, observation)[0])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def state_process(state):
    #state: (player points, banker points, usable ace)
    state_nor = []
    for i in range(len(state)):
        if i == 0:
            state_nor.append(state[i]/32.0)
        elif i == 1:
            state_nor.append(state[i]/11.0)
        else:
            if state[i] == True:
                state_nor.append(1.0)
            else:
                state_nor.append(0.0)
    state_array = np.array(state_nor).reshape(1,3)
    return state_array

def td_network(
    env,
    sess,
    estimator,
    episode_num=10000,
    discount_factor=0.9,
    epsilon_max=0.1,
    epsilon_min=0.0001
    ):
    epsilons = np.linspace(epsilon_max, epsilon_min, episode_num)
    policy = make_epsilon_greedy_policy(estimator, env.nA)

    for i_episode in range(episode_num):
        state = env.reset()
        state_array = state_process(state)
        done = False
        epsilon = epsilons[i_episode]

        A = policy(sess, state_array, epsilon)
        action = np.random.choice(np.arange(env.nA), p=A)
        action_array = np.array(action).reshape(1)

        while not done:
            next_state, reward, done, info = env.step(action)
            next_state_array = state_process(next_state)

            if done:
                td_target = reward + discount_factor * 0.0
                td_target_array = np.array(td_target).reshape(1)

                print('episode:{}'.format(i_episode))
                print('action:{}'.format(action))
                print('done:{}'.format(done))
                print('td target:{}'.format(td_target))
                print('reward:{}'.format(reward))
                print('state:{}'.format(state))
                print('normalization of state:{}'.format(state_array))
                print('='*20)

                estimator.update(sess, state_array, action_array, td_target_array)
                break

            else:
                next_A = policy(sess, next_state_array, epsilon)
                next_action = np.random.choice(np.arange(env.nA), p=next_A)
                next_action_array = np.array(next_action).reshape(1)
                next_qs = estimator.predict(sess, next_state_array)[0]
                next_q = next_qs[next_action]

                td_target = reward + discount_factor * next_q
                td_target_array = np.array(td_target).reshape(1)

                print('episode:{}'.format(i_episode))
                print('action:{}'.format(action))
                print('done:{}'.format(done))
                print('td target:{}'.format(td_target))
                print('reward:{}'.format(reward))
                print('state:{}'.format(state))
                print('normalization of state:{}'.format(state_array))
                print('='*20)

                estimator.update(sess, state_array, action_array, td_target_array)

                state = next_state
                state_array = state_process(state)
                action = next_action
                action_array = np.array(action).reshape(1)

    V = defaultdict(float)
    all_state = []
    for i in range(11, 22):
        for j in range(1, 11):
            for k in [True, False]:
                all_state.append((i, j, k))
    for state in all_state:
        nor_state = state_process(state)
        v_ = np.max(estimator.predict(sess, nor_state))
        V[state] = v_
    print('V:{}'.format(V))
    return V

def td_network_test():
    env = BlackjackEnv()
    estimator = Estimator(learning_rate=0.003)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        V = td_network(env, sess, estimator)
        #print(sess.run(estimator.w))
        #print(sess.run(estimator.b))
    plotting.plot_value_function(V, title='Optimal Value')

def main():
    td_network_test()

if __name__ == '__main__':
    main()























