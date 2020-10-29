from Blackjack_ import BlackjackEnv
from collections import defaultdict
import plotting
import numpy as np
import sys
import tensorflow as tf

#epsilon-greedy(Q)
def epsilon_greedy_policy(Q, observation, nA, epsilon):
    best_action = np.argmax(Q[observation])
    A = np.ones(nA, dtype=np.float) * epsilon /nA
    A[best_action] += 1-epsilon
    return A

#MC Control with GLIE
#use N(s, a) not alpha
def mc_control_with_epsilon_greedy(
    env,
    episode_nums,   #number of episodes
    discount_factor = 1.0,  #discount factor gamma
    epsilon_max = 0.1,  #max of epsilon
    epsilon_min = 0.0001    #min of epsilon
    ):
    #epilon list with gradually decreasing epsilon
    epsilon = np.linspace(epsilon_max, epsilon_min,episode_nums)
    #Q[state][action]
    Q = defaultdict(lambda:np.zeros(env.nA))
    #N[(state, action)]
    return_count = defaultdict(float)

    #sample
    for i_episode in range(1, 1+episode_nums):
        #state: (player's points, shown dealer point, usable ace)
        state = env.reset()
        #current episode: [(S, A, R)]
        episode = []
        done = False
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}.'.format(i_episode, episode_nums))
            sys.stdout.flush()

        #make sure this episode comes to terminal
        while not done:
            A = epsilon_greedy_policy(Q, state, env.nA, epsilon[i_episode - 1])
            probs = A
            action = np.random.choice(np.arange(env.nA), p=probs)
            #take action
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        #get S, A from whole episode
        seperate_episode = set([(tuple(x[0]), x[1]) for x in episode])

        #update Q[state][action]
        for state,action in seperate_episode:
            #idx: index, e:(state, action, reward)
            for idx,e in enumerate(episode):
                if e[0]==state and e[1]==action:
                    #first visit MC
                    first_visit_idx = idx
                    break
            pair = (state,action)
            #i: time-step, e:(state, action, reward)
            G = sum([e[2]*(discount_factor**i) for i,e in enumerate(episode[first_visit_idx:])])
            '''
            G = 0.0
            for i, e in enumerate(episode[first_visit_idx:]):
                G += (e[2] * (discount_factor**i)
            '''
            #N(s, a) += 1
            return_count[pair] += 1.0
            #update Q with G
            Q[state][action] += ((G * 1.0 - Q[state][action])/return_count[pair])
    return Q

#test for MC_Control_with_epsilon_greedy
def mc_control_with_epsilon_greedy_test():
    env = BlackjackEnv()
    Q = mc_control_with_epsilon_greedy(env, episode_nums=10000)
    V = defaultdict(float)
    for state, actions in Q.items():
        max_q = np.max(actions)
        V[state] = max_q
    plotting.plot_value_function(V, title='Optimal Value Function')

#SARSA Control with fixed alpha
def sarsa(
    env,
    episode_nums,
    discount_factor = 1.0,
    alpha = 0.1,
    epsilon_max = 0.1,
    epsilon_min = 0.0001
    ):
    epsilon = np.linspace(epsilon_max, epsilon_min, episode_nums)
    Q = defaultdict(lambda:np.zeros(env.nA))

    #sample
    for i_episode in range(1, 1+episode_nums):
        state = env.reset()
        done = False
        A = epsilon_greedy_policy(Q, state, env.nA, epsilon[i_episode - 1])
        probs = A
        action = np.random.choice(np.arange(env.nA), p=probs)

        #print info
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}.'.format(i_episode, episode_nums))
            sys.stdout.flush()

        #sample and update Q
        while not done:
            new_state, reward, done, info = env.step(action)
            if done:
                Q[state][action] += (alpha * (reward + discount_factor * 0.0 - Q[state][action]))
                break
            else:
                new_A = epsilon_greedy_policy(Q, new_state, env.nA, epsilon[i_episode - 1])
                probs = new_A
                new_action = np.random.choice(np.arange(env.nA), p=probs)
                Q[state][action] += (alpha * (reward + discount_factor * Q[new_state][new_action] - Q[state][action]))
                state = new_state
                action = new_action
    return Q

#test for sarsa
def sarsa_test():
    env = BlackjackEnv()
    Q = sarsa(env, episode_nums=10000)
    V = defaultdict(float)
    for state, actions in Q.items():
        max_q = np.max(actions)
        V[state] = max_q
    plotting.plot_value_function(V, title='Optimal Value Function')

#SARSA(lambda) Control with fixed alpha
def sarsa_lambda(
    env,
    episode_nums,
    discount_factor = 1.0,
    alpha = 0.1,
    epsilon_max = 0.1,
    epsilon_min = 0.0001,
    lambda_ = 0.5
    ):
    epsilon = np.linspace(epsilon_max, epsilon_min, episode_nums)
    Q = defaultdict(lambda:np.zeros(env.nA))

    #sample
    for i_episode in range(1, 1+episode_nums):
        E = defaultdict(lambda:np.zeros(env.nA))
        states = []
        state = env.reset()
        states.append(state)
        done = False
        A = epsilon_greedy_policy(Q, state, env.nA, epsilon[i_episode - 1])
        probs = A
        action = np.random.choice(np.arange(env.nA), p=probs)

        #print info
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}.'.format(i_episode, episode_nums))
            sys.stdout.flush()

        #sample and update Q
        while not done:
            new_state, reward, done, info = env.step(action)
            states.append(new_state)
            states = list(set(states))
            if done:
                Q[state][action] += (alpha * (reward + discount_factor * 0.0 - Q[state][action]))
                break
            else:
                new_A = epsilon_greedy_policy(Q, new_state, env.nA, epsilon[i_episode - 1])
                probs = new_A
                new_action = np.random.choice(np.arange(env.nA), p=probs)
                delta = reward + discount_factor * Q[new_state][new_action] - Q[state][action]
                E[state][action] += 1.0
                for every_s in states:
                    for every_a in range(env.nA):
                        Q[every_s][every_a] = Q[every_s][every_a] + alpha * delta * E[every_s][every_a]
                        E[every_s][every_a] = discount_factor * lambda_ * E[every_s][every_a]
                state = new_state
                action = new_action
    return Q

#test for sarsa_lambda
def sarsa_lambda_test():
    env = BlackjackEnv()
    Q = sarsa_lambda(env, episode_nums=10000)
    V = defaultdict(float)
    for state, actions in Q.items():
        max_q = np.max(actions)
        V[state] = max_q
    plotting.plot_value_function(V, title='Optimal Value Function')

#q-learning
def q_learning(
    env,
    episode_nums,
    discount_factor = 1.0,
    alpha = 0.5,
    epsilon_max = 0.1,
    epsilon_min = 0.0001,
    ):
    epsilon = np.linspace(epsilon_max, epsilon_min, episode_nums)
    Q = defaultdict(lambda:np.zeros(env.nA))

    #sample
    for i_episode in range(1, 1+episode_nums):
        state = env.reset()
        done = False
        A = epsilon_greedy_policy(Q, state, env.nA, epsilon[i_episode-1])
        probs = A
        action = np.random.choice(np.arange(env.nA), p=probs)

        #pring info
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}.'.format(i_episode, episode_nums))

        #sample
        while not done:
            new_state, reward, done, info = env.step(action)
            if done:
                Q[state][action] += (alpha * (reward + discount_factor * 0.0 - Q[state][action]))
                break
            else:
                greedy_new_A = epsilon_greedy_policy(Q, new_state, env.nA, 0)
                probs = greedy_new_A
                greedy_new_action = np.random.choice(np.arange(env.nA), p=probs)
                Q[state][action] += (alpha * (reward + discount_factor * Q[new_state][greedy_new_action] - Q[state][action]))
                new_A = epsilon_greedy_policy(Q, new_state, env.nA, epsilon[i_episode-1])
                probs = new_A
                new_action = np.random.choice(np.arange(env.nA), p=probs)
                state = new_state
                action = new_action
    return Q

#test for q_learning
def q_learning_test():
    env = BlackjackEnv()
    Q = q_learning(env, episode_nums=10000)
    V = defaultdict(float)
    for state, actions in Q.items():
        max_q = np.max(actions)
        V[state] = max_q
    plotting.plot_value_function(V, title='Optimal Value Function')

#TD FA
def td_fa(
    env,
    sess,
    estimator,
    episode_num=1000,
    discount_factor=0.99,
    epsilon_max=0.1,
    epsilon_min=0.0001,
    ):
    epsilons = np.linspace(epsilon_max, epsilon_min, episode_num)
    policy = make_epsilon_greedy_policy(estimator, env.nA)
    for i_episode in range(episode_num):
        state = env.reset()
        state_array = state_process(state)
        done = False
        while not done:
            A = policy(sess, state_array, epsilons[i_episode])
            action = np.random.choice(np.arange(env.nA), p=A)
            next_state, reward, done, info = env.step(action)
            next_state_array = state_process(next_state)
            if done:
                td_target = reward + discount_factor * 0.0
                td_target_array = np.array(td_target).reshape(1)
                action_array = np.array(action).reshape(1)
                estimator.update(sess, state_array, action_array, td_target_array)
                break
            else:
                next_q = estimator.predict(sess, next_state_array)
                td_target = reward + discount_factor * np.max(next_q)
                td_target_array = np.array(td_target).reshape(1)
                action_array = np.array(action).reshape(1)
                estimator.update(sess, state_array, action_array, td_target_array)
                state = next_state
        print('episode:{}/{}'.format(i_episode, episode_num))

#q_value estimator
class Estimator(object):
    def __init__(self):
        self._build_model()
    def _build_model(self):
        n_hidden_1 = 32
        n_hidden_2 = 32
        n_input = 3
        n_class = 2
        self.x_pl = tf.placeholder(tf.float32, [None, n_input])
        self.y_pl = tf.placeholder(tf.float32, [None])
        self.actions_pl = tf.placeholder(tf.int32, [None])
        batch_size  = tf.shape(self.x_pl)[0]
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
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        #loss, optimizer, init
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_predictions, labels=self.y_pl)
        self.loss = tf.reduce_mean(self.losses)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def predict(self, sess, s):
        return sess.run(self.predictions, {self.x_pl:s})
    
    def update(self, sess, s, a, y):
        _ = sess.run(self.train_op, feed_dict={self.x_pl:s, self.y_pl:y, self.actions_pl:a})

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, observation)[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def state_process(state):
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

def td_fa_test():
    env = BlackjackEnv()
    estimator = Estimator()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        td_fa(env, sess, estimator)
        sess.close()


def main():
    #mc_control_with_epsilon_greedy_test()
    #sarsa_test()
    #sarsa_lambda_test()
    q_learning_test()
    #td_fa_test()

if __name__ == '__main__':
    main()






























