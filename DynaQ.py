import tensorflow as tf
import numpy as np
from Blackjack_ import BlackjackEnv
from collections import defaultdict
import plotting
import time

#dyna_q
def dyna_q(
    env,
    sess,
    estimator,
    model,
    episode_num=3000,
    train_model_times=1000,      #sample 1000 for model training
    train_with_model_times=5,         #train estimator with model 10 times after actual env. train
    discount_factor=0.9,
    epsilon_max=0.1,
    epsilon_min=0.0001
    ):
    #epsilons: decay while sampling
    epsilons = np.linspace(epsilon_max, epsilon_min, episode_num)
    policy = make_epsilon_greedy_policy(estimator, env.nA)

    #sample 1000 data for model initial construction
    print('Sample for model...')
    for _ in range(train_model_times):
        state = env.reset()
        state_array = state_process(state)
        done = False
        while not done:
            action = np.random.choice(np.arange(env.nA))
            next_state, reward, done, info = env.step(action)
            next_state_array = state_process(next_state)
            #s_a: [1 * 4], [state, action]
            s_a = np.append(state_array, float(action)).reshape((1,4))
            #next_s_r: [1 * 4], [s', r]
            next_s_r = np.append(next_state_array, float(reward)).reshape((1,4))
            '''
            print('state:{}'.format(state))   
            print('action:{}'.format(action))
            print('s_a:{}'.format(s_a))
            print('next_state:{}'.format(next_state))
            print('reward:{}'.format(reward))
            print('next_s_r:{}'.format(next_s_r))
            print('predictions:{}'.format(predictions))
            print('-'*50)
            '''
            model.update(sess, s_a, next_s_r)
            if done:
                break
            state = next_state
            state_array = state_process(state)
            time.sleep(0.01)

    #sample in actual Env and Model
    #update estimator with data from Env and Model
    #update Model with actual data from Env
    print('Here is DynaQ...')
    print('='*50)
    for i_episode in range(episode_num):
        state = env.reset()
        #process state
        state_array = state_process(state)
        #initialization for done
        done = False
        print('episode:{}/{}'.format(i_episode, episode_num))
        while not done:
            #epsilon-greedy policy
            A = policy(sess, state_array, epsilons[i_episode])
            #get action
            action = np.random.choice(np.arange(env.nA), p=A)
            action_array = np.array(action).reshape(1)
            #take action
            next_state, reward, done, info = env.step(action)
            #state process
            next_state_array = state_process(next_state)
            #next_q
            next_q = estimator.predict(sess, next_state_array)
            #get best action for next_state
            best_action = np.argmax(next_q, axis=1)
            #target = r + done * gamma * max(q(s', a))
            target = reward + np.invert(done).astype(np.float32) * discount_factor * next_q[0, best_action]
            target_array = np.array(target).reshape(1)

            '''
            print('action:{}'.format(action))
            print('done:{}'.format(done))            
            print('q target:{}'.format(target))
            print('predictions:{}'.format(estimator.predict(sess, state_array)))
            print('reward:{}'.format(reward))
            print('state:{}'.format(state))
            print('normalization of state:{}'.format(state_array))
            '''
            time.sleep(0.01)
            #update estimator
            estimator.update(sess, state_array, action_array, target_array)

            #update model
            s_a = np.append(state_array, float(action)).reshape((1,4))
            next_s_r = np.append(next_state_array, float(reward)).reshape((1,4))
            model.update(sess, s_a, next_s_r)

            if done:
                break

            #s = s'
            state = next_state
            state_array = state_process(state)

        #update estimator with model
        for i in range(train_with_model_times):
            #state: random state
            state = np.random.random((1,3))
            state_array = state_process(state)
            #random action from state
            action = np.random.choice(np.arange(env.nA))
            action_array = np.array(action).reshape(1)
            #s_a
            s_a = np.append(state_array, float(action)).reshape((1,4))
            #predictions: [[s', r]]
            next_s_r = model.predict(sess, s_a)[0]
            #next_state_array: [[s']]
            next_state_array = next_s_r[:3].reshape(1,3)
            #reward
            reward = next_s_r[-1]
            #*********************
            #here we treat s' is no terminal
            #it's a question whether to promote with s' is terminal so that q(s') is 0
            #*********************

            #get q(s')
            next_q = estimator.predict(sess, next_state_array)
            #best action
            best_action = np.argmax(next_q, axis=1)
            #target for estimator, same with target in Env.
            #target: [target]
            target = reward + discount_factor * next_q[0, best_action]
            #reshape or not, both are ok
            target_array = np.array(target).reshape(1)
            '''
            print('state:{}'.format(state))
            print('action:{}'.format(action))
            print('s_a:{}'.format(s_a))
            print('next_s_r predicted by model:{}'.format(next_s_r))
            print('next_state_array:{}'.format(next_state_array))
            print('reward:{}'.format(reward))
            print('next_q:{}'.format(next_q))
            print('target_array:{}'.format(target_array))
            print('='*20)
            '''
            #sleep for not crash
            time.sleep(0.01)
            #update estimator with model
            estimator.update(sess, state_array, action_array, target_array)


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

#model
class Model(object):
    def __init__(self, learning_rate=0.003):
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        n_hidden_1 = 8
        n_hidden_2 = 8
        #input: [s, a]
        n_input = 4
        #output: [s', r]
        n_class = 4

        self.x_pl = tf.placeholder(tf.float32, [None, n_input])
        self.y_pl = tf.placeholder(tf.float32, [None, n_class])
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
        layer_1 = tf.add(tf.matmul(self.x_pl, self.w['h1']), self.b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.w['h2']), self.b['b2'])
        layer_2 = tf.nn.relu(layer_2)
        self.predictions = tf.add(tf.matmul(layer_2, self.w['out']), self.b['out'])
        self.loss = tf.reduce_mean(tf.square(self.predictions - self.y_pl))    
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, s_a):
        return sess.run(self.predictions, feed_dict={self.x_pl:s_a})

    def update(self, sess, s_a, next_s_r):
        _ = sess.run(self.train_op, feed_dict={self.x_pl:s_a, self.y_pl:next_s_r})

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

def dyna_q_test():
    env = BlackjackEnv()
    estimator = Estimator(0.003)
    model = Model(0.003)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        V = dyna_q(env, sess, estimator, model, episode_num=3000, train_model_times=3000, train_with_model_times=3)
    plotting.plot_value_function(V, title='Optimal Value Function') 

def main():
    dyna_q_test()

if __name__ == '__main__':
    main()


