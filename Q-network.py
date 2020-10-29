import tensorflow as tf
import numpy as np
from Blackjack_ import BlackjackEnv
from collections import defaultdict
import plotting
import time

#q network
def q_network(
    env,
    sess,
    estimator,
    episode_num=3000,
    discount_factor=0.9,
    epsilon_max=0.1,
    epsilon_min=0.0001
    ):
    #epsilons: decay while sampling
    epsilons = np.linspace(epsilon_max, epsilon_min, episode_num)
    policy = make_epsilon_greedy_policy(estimator, env.nA)

    #sample
    for i_episode in range(episode_num):
        state = env.reset()
        #process state
        state_array = state_process(state)
        #initialization for done
        done = False
        while not done:
            #epsilon-greedy policy
            A = policy(sess, state_array, epsilons[i_episode])
            #get action
            action = np.random.choice(np.arange(env.nA), p=A)
            #take action
            next_state, reward, done, info = env.step(action)
            #state process
            next_state_array = state_process(next_state)
            #if done, q(next_s, a') = 0
            if done:
                #target = r + gamma * q(next_s, a')
                q_target = reward + discount_factor * 0.0
                #reshape for tf.train_op
                q_target_array = np.array(q_target).reshape(1)
                action_array = np.array(action).reshape(1)
                print('episode:{}'.format(i_episode))
                print('action:{}'.format(action))
                print('done:{}'.format(done))
                print('q target:{}'.format(q_target))
                print('predictions:{}'.format(estimator.predict(sess, state_array)))
                print('reward:{}'.format(reward))
                print('state:{}'.format(state))
                print('normalization of state:{}'.format(state_array))
                print('='*20)
                #update estimator
                estimator.update(sess, state_array, action_array, q_target_array)
                break
            else:
                #self.predictions: [[q(s, a_1), q(s, a_2)]]
                next_q = estimator.predict(sess, next_state_array)[0]
                #q_target = r + gamma * max(q(next_s))
                q_target = reward + discount_factor * np.max(next_q)
                q_target_array = np.array(q_target).reshape(1)
                action_array = np.array(action).reshape(1)
                print('episode:{}'.format(i_episode))
                print('action:{}'.format(action))
                print('done:{}'.format(done))                
                print('q target:{}'.format(q_target))
                print('predictions:{}'.format(estimator.predict(sess, state_array)))
                print('reward:{}'.format(reward))
                print('state:{}'.format(state))
                print('normalization of state:{}'.format(state_array))
                print('='*20)
                #update estimator
                estimator.update(sess, state_array, action_array, q_target_array)
                #s = next_s
                state = next_state
    
    #get all state and v(s)
    #v(s) is set with max(q(s))
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
    #print('V:{}'.format(V))
    return V

#q_value estimator
class Estimator(object):

    def __init__(self, learning_rate=0.003):
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        #3 layers
        n_hidden_1 = 8
        n_hidden_2 = 8
        n_input = 3
        n_class = 2

        #training params
        self.x_pl = tf.placeholder(tf.float32, [None, n_input])
        self.y_pl = tf.placeholder(tf.float32, [None])
        self.actions_pl = tf.placeholder(tf.int32, [None])
        batch_size = tf.shape(self.x_pl)[0]
        self.w = {
            'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_hidden_2, n_class]))
        }
        self.b = {
            'b1':tf.Variable(tf.random_normal([n_hidden_1])),
            'b2':tf.Variable(tf.random_normal([n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_class]))
        }

        #layers
        layer_1 = tf.add(tf.matmul(self.x_pl, self.w['h1']), self.b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.w['h2']), self.b['b2'])
        layer_2 = tf.nn.relu(layer_2)
        self.predictions = tf.add(tf.matmul(layer_2, self.w['out']), self.b['out'])

        #get predictions under actions just taken
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        #loss, optimizer, init
        #self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_predictions, labels=self.y_pl)
        #**************
        #use MSE not cross_entropy for better training performance
        #weights are not be updated when use cross entropy as loss fn.
        #**************
        self.losses = tf.square(self.action_predictions - self.y_pl)
        self.loss = tf.reduce_mean(self.losses)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def predict(self, sess, s):
        #get predictions
        #self.predictions: [[q(s, a_1), q(s, a_2)]]
        return sess.run(self.predictions, {self.x_pl:s})
    
    def update(self, sess, s, a, y):
        #update estimator params
        _ = sess.run(self.train_op, feed_dict={self.x_pl:s, self.y_pl:y, self.actions_pl:a})

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #q_values: [q(s, a_1), q(s, a_2)]
        q_values = estimator.predict(sess, observation)[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

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

def q_network_test():
    env = BlackjackEnv()
    estimator = Estimator(0.001)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        V = q_network(env, sess, estimator, episode_num=10000)
    plotting.plot_value_function(V, title='Optimal Value Function')    

def main():
    start = time.clock()
    q_network_test()
    print(time.clock() - start)

if __name__ == '__main__':
    main()


