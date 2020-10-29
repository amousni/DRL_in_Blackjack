import tensorflow as tf
import numpy as np
from Blackjack_ import BlackjackEnv
from collections import defaultdict, deque, namedtuple
import plotting
import itertools
import random
import os
import sys

#q_value estimator
class Estimator(object):
    #initialization
    def __init__(self, learning_rate=0.003, scope='estimator', summaries_dir=None):
        self.learning_rate = learning_rate
        #name scope
        self.scope = scope
        #summary writer
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            #have summaries dir
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

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

        #summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.histogram('loss_hist', self.losses),
            tf.summary.histogram('q_values_hist', self.predictions),
            tf.summary.scalar('max_q_value', tf.reduce_max(self.predictions))
            ])
    
    def predict(self, sess, s):
        #get predictions
        #self.predictions: [[q(s, a_1), q(s, a_2)]]
        return sess.run(self.predictions, {self.x_pl:s})
    
    def update(self, sess, s, a, y):
        #update estimator params
        feed_dict = {self.x_pl:s, self.y_pl:y, self.actions_pl:a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict=feed_dict
            )
        #add summary
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    #each parameter takes an action
    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        #assign e2_v with e1_v
        op = e2_v.assign(e1_v)
        update_ops.append(op)
    sess.run(op)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #q_values: [[q(s, a_1), q(s, a_2)]]
        q_values = estimator.predict(sess, observation)
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
    #state_array: [[player points, banker points, usable ace]]
    return state_array

#q network
def q_network(
    env,                                    #env
    sess,                                   #sess
    q_estimator,                            #network to estimate actual q value
    target_estimator,                       #old network to predict target
    experiment_dir,                         #summary dir
    replay_memory_size=10000,               #max volumn of replay memory pool
    replay_memory_init_size=1000,           #initial volumn of replay memory which need sample first
    update_target_estimator_every=200,      #update target netword every 200 action
    batch_size=20,                          #batch size for one-time train
    episode_num=1000,                        #train episode
    discount_factor=0.9,                    #gamma
    epsilon_max=0.1,                        #epsilon max
    epsilon_min=0.0001                      #epsilon min, decay with episode num
    ):

    #Transition, one piece of data: [s, a, r, s', done]
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    #Data set
    replay_memory = []
    #statics for each episode
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    stats = EpisodeStats(episode_lengths=np.zeros(episode_num), episode_rewards=np.zeros(episode_num))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    #path and dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()

    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    #action step, every 200 actions, update target network
    total_t = sess.run(tf.contrib.framework.get_global_step())

    #epsilons: decay while sampling
    epsilons = np.linspace(epsilon_max, epsilon_min, episode_num)
    policy = make_epsilon_greedy_policy(q_estimator, env.nA)

    print('='*20)
    print('sample for replay memory')
    state = env.reset()
    state_array = state_process(state)

    #get initial dataset
    for i in range(replay_memory_init_size):
        #dataset initialization with epsilons[0]
        A = policy(sess, state_array, epsilons[0])
        action = np.random.choice(np.arange(env.nA), p=A)
        next_state, reward, done, info = env.step(action)
        next_state_array = state_process(next_state)
        #state_array: 1*3 np.array, [[player points, banker points, usable ace]]
        #action: int, 0 for stick, 1 for call
        #reward: float, -1, 0, 1
        #next_state_array: same for state_array
        replay_memory.append(Transition(state_array, action, reward, next_state_array, done))
        if done:
            state = env.reset()
            state_array = state_process(state)
        else:
            state = next_state
            state_array = state_process(state)

    #sample
    for i_episode in range(episode_num):

        #save checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        #reset env
        state = env.reset()
        #process state
        state_array = state_process(state)
        #epsilon
        epsilon = epsilons[i_episode]

        #t: steps of the episode
        for t in itertools.count():

            #add epsilon to tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag='epsilon')
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            #update target estimator
            if total_t % update_target_estimator_every == 0 and total_t != 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print('\nCopied model parameters to target network.')
            sys.stdout.flush()

            #epsilon-greedy policy
            A = policy(sess, state_array, epsilon)
            #get action
            action = np.random.choice(np.arange(env.nA), p=A)
            #take action
            next_state, reward, done, info = env.step(action)
            #state process
            next_state_array = state_process(next_state)

            #pop the oldest data if dataset is full
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            #add new data to dataset
            replay_memory.append(Transition(state_array, action, reward, next_state_array, done))

            #update statisitc
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            #sample a minibatch from replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            #states_batch: [batch_size * 1 * len(state)], same for next_states_batch, process two state batch
            #sess.run(tf.squeeze(states_batch))
            states_batch, next_states_batch = states_batch.reshape((batch_size, len(state))), next_states_batch.reshape((batch_size, len(state)))

            #calculate q and targets
            q_values_next = q_estimator.predict(sess, next_states_batch)
            #***********************************
            #!!!best actions from q estimator!!!
            #!target value from target estimator
            #***********************************
            best_actions = np.argmax(q_values_next, axis=1)
            target_values = target_estimator.predict(sess, next_states_batch)
            """
            #some infos
            print('reward_batch:{}'.format(reward_batch))
            print('done_batch:{}'.format(done_batch))
            print('target_values:{}'.format(target_values))
            print('best_actions:{}'.format(best_actions))
            """
            #target = r + done * gamma * target_q(s', a' from q_estimator)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * target_values[np.arange(batch_size), best_actions]

            #update
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            
            #action step += 1
            #action step could be inverted to episode step
            #in Blackjack, taking one action then come to terminal is common
            #for copy q network to target network effectively, total_t update every action
            total_t += 1

            #if done, episode break
            if done:
                print('episode: {}/{}'.format(i_episode, episode_num))
                print('='*70)
                break

            #s = s'
            state = next_state
            state_array = state_process(state)

        #add summaries to tensorboard
        episode_summary = tf.Summary()
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_rewards", tag="episode_rewards")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_lengths", tag="episode_lengths")

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
        v_ = np.max(q_estimator.predict(sess, nor_state))
        V[state] = v_
    #print('V:{}'.format(V))
    return V

def q_network_test(
    batch_size=20,
    episode_num=3000,
    replay_memory_init_size=1000):
    
    tf.reset_default_graph()

    #experiment directory
    experiment_dir = os.path.abspath('./experiment/{}'.format('Blackjack'))
    
    #global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    #q_estimator and target estimator
    q_estimator = Estimator(scope='q', summaries_dir=experiment_dir)
    target_estimator = Estimator(scope='target_q')
    
    #env
    env = BlackjackEnv()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        V = q_network(
            env,
            sess,
            q_estimator=q_estimator, 
            target_estimator=target_estimator, 
            experiment_dir=experiment_dir,
            episode_num=episode_num,
            batch_size=batch_size,
            replay_memory_init_size=replay_memory_init_size
            )
    plotting.plot_value_function(V, title='Optimal Value Function')    

def main():
    q_network_test(
        batch_size=30,
        episode_num=100,
        replay_memory_init_size=300)

if __name__ == '__main__':
    main()


