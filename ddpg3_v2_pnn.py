import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
import os
import time
import pickle
# import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
import envs


SCALE = 1.
# ===========================
#   Actor and Critic DNNs
# ===========================
def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape, mean = 0,stddev = 0.01))
def bias_init(shape):
    return tf.Variable(tf.constant(0.01, shape = shape))
def uniform_init(shape):
    return tf.Variable(tf.random_uniform(shape, minval = -0.003, maxval = 0.003))

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, weights,
                    cascade_net):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network(weights, cascade_net)

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = \
                            self.create_actor_network(weights, cascade_net, target = True)

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)
    def create_actor_network(self, weights, cascade_net, target = False):
        inputs = tf.placeholder(tf.float32, shape = [None, self.s_dim])

        # new newwork
        w1, b1 = weights_init([self.s_dim, 400]), bias_init([400])
        if cascade_net:
            w1_p,b1_p,scale1 = weights_init([self.s_dim,400]),bias_init([400]),tf.Variable(SCALE)
        else:
            w1_p ,b1_p, scale1 = tf.constant(weights['w1']), tf.constant(weights['b1']), tf.Variable(SCALE)

        layer1_p = tf.nn.relu(tf.matmul(inputs, w1_p) + b1_p)
        layer1 = tf.nn.relu(tf.matmul(inputs, w1) + b1) + layer1_p * scale1


        w2, b2 = weights_init([400, 300]), bias_init([300])
        if cascade_net:
            w2_p,b2_p,scale2 = weights_init([400,300]),bias_init([300]),tf.Variable(SCALE)
        else:
            w2_p ,b2_p, scale2 = tf.constant(weights['w2']), tf.constant(weights['b2']), tf.Variable(SCALE)
        layer2_p = tf.nn.relu(tf.matmul(layer1_p, w2_p) + b2_p)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2) + layer2_p * scale2
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w3, b3 = uniform_init([300, self.a_dim]), uniform_init([self.a_dim])
        out = tf.nn.tanh(tf.matmul(layer2, w3) + b3)
       
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    def save_weights(self):
        for key, value in self.weights.items():
            self.weights[key] = self.sess.run(value)
        save_dir = './results/weights'
        file_name = os.path.join(save_dir, 'actor_weights.dat')
        file = open(file_name, 'wb')
        pickle.dump(self.weights, file, 1)




class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, weights,
                        cascade_net):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.weights = {}

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network(weights,cascade_net)

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = \
                            self.create_critic_network(weights, cascade_net,target = True)

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, weights, cascade_net, target = False):
        inputs = tf.placeholder(tf.float32, shape = [None, self.s_dim])
        action = tf.placeholder(tf.float32, shape = [None, self.a_dim])

        w1, b1 = weights_init([self.s_dim, 200]), bias_init([200])
        if cascade_net:
            w1_p,b1_p,scale1 = weights_init([self.s_dim,200]),bias_init([200]),tf.Variable(SCALE)
        else:
            w1_p, b1_p, scale1 = tf.constant(weights['w1']), tf.constant(weights['b1']), tf.Variable(SCALE)
        layer1_p = tf.nn.relu(tf.matmul(inputs, w1_p) + b1_p)
        layer1 = tf.nn.relu(tf.matmul(inputs, w1) + b1) + layer1_p * scale1

        w2, b2 = weights_init([200, 100]), bias_init([100])
        w2_action = weights_init([self.a_dim, 100])
        if cascade_net:
            w2_p,b2_p,scale2 = weights_init([200,100]),bias_init([100]),tf.Variable(SCALE)
            w2_action_p = weights_init([self.a_dim,100])
        else:
            w2_p, b2_p, scale2 = tf.constant(weights['w2']), tf.constant(weights['b2']), tf.Variable(SCALE)
            w2_action_p = tf.constant(weights['w2_action'])
        layer2_p = tf.nn.relu(tf.matmul(layer1, w2_p) + tf.matmul(action, w2_action_p) + b2_p)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + tf.matmul(action, w2_action) + b2) + layer2_p *scale2

        w3, b3 = uniform_init([100, 1]), uniform_init([1])
        out = tf.matmul(layer2, w3) + b3


        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    def save_weights(self):
        for key, value in self.weights.items():
            self.weights[key] = self.sess.run(value)
        save_dir = './results/weights'
        file_name = os.path.join(save_dir, 'critic_weights.dat')
        file = open(file_name, 'wb')
        pickle.dump(self.weights, file, 1)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta= 0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # self.sigma-=DELTA
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    def eval_reward(env,actor,max_episode_len,episode_i):
        #evaluate actor network without noise
        ep_num = 5
        ep_reward = 0
        for i in range(ep_num):
            # s=env.reset_to_value(rad_unit*i)
            s = env.reset()
            for k in range(max_episode_len):
                a=actor.predict_target(np.reshape(s, (1, actor.s_dim)))
                s2, r, terminal = env.step(a[0])
                ep_reward += r
                if terminal:
                    break
                s=s2
        ep_reward //= ep_num
        # print('Episodic Reward: %d, Elapsed time: %.4f' % (int(ep_reward),elapsed))
        print('episode: %d,Episodic Reward: %d' % (episode_i, ep_reward))
        return ep_reward
    def save_reward(lst,args):
        base_dir = args['rewards_dir']
        time_stamp=time.strftime('%m%d-%H%M%S')
        base_dir = os.path.join(base_dir,time_stamp)
        os.makedirs(base_dir,exist_ok=1)
        save_file_name=os.path.join(base_dir,'rwd.dat')
        file=open(save_file_name,'wb')
        pickle.dump(lst,file,1)
        # plt.plot(lst)
        # plt.title(time_stamp)
        # plt.xlabel('Episodes')
        # plt.ylabel('Average Reward')
        # plt.ylim([-300,0])
        fig_name = os.path.join(base_dir, 'reward_fig.png')
        # plt.savefig(fig_name)
        print('Rewards sucessfully writed!')


    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    reward_list=[]
    saver = tf.train.Saver()
    max_eval_rwd = -10000

    for i in range(int(args['max_episodes'])):
        
        s = env.reset()
        

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                break
        eval_r = eval_reward(env,actor,int(args['max_episode_len']),i)
        reward_list.append(eval_r)
    save_reward(reward_list,args)

def random_weights(s_dim, a_dim):
    actor_weights = {}
    critic_weights = {}
    std1, std2 = 0.1,0.1
    actor_weights['w1'] = (-0.014 + np.random.randn(s_dim,400) * 0.16).astype(np.float32) # make weights fall in (-0.5,0.5)
    actor_weights['w2'] = (-0.023 + np.random.randn(400,300) * 0.036).astype(np.float32) # make weights fall in (-0.2,0.2)
    actor_weights['b1'] = (-0.037 + np.random.randn(400) * 0.083).astype(np.float32)
    actor_weights['b2'] = (-0.017 + np.random.randn(300) * 0.049).astype(np.float32)

    critic_weights['w1'] = (0.015 + np.random.randn(s_dim,200) * 0.209).astype(np.float32)
    critic_weights['b1'] = (0.029 + np.random.randn(200) * 0.131).astype(np.float32)
    critic_weights['w2'] = (0.001 + np.random.randn(200,100) * 0.196).astype(np.float32)
    critic_weights['b2'] = (0.072 + np.random.randn(100) * 0.114).astype(np.float32)
    critic_weights['w2_action'] = (0.009 + np.random.randn(a_dim,100) * 0.096).astype(np.float32)
    return actor_weights,critic_weights
def main(args):

    with tf.Session() as sess:

        env = envs.ArmEnvShort()
        # np.random.seed(int(args['random_seed']))
        # tf.set_random_seed(int(args['random_seed']))

        state_dim = env.state_dim
        action_dim = env.action_dim
        action_bound = max(env.action_bound)

        print(args['rand_init'])
        if not args['rand_init']:
            weights_save_dir = './results/weights'
            actor_file_name = os.path.join(weights_save_dir, 'actor_weights.dat')
            critic_file_name = os.path.join(weights_save_dir, 'critic_weights.dat')
            actor_file = open(actor_file_name, 'rb')
            critic_file = open(critic_file_name, 'rb')
            actor_weights = pickle.load(actor_file)
            critic_weights = pickle.load(critic_file)
            actor_file.close()
            critic_file.close()
        else:
            actor_weights, critic_weights = random_weights(env.state_dim, env.action_dim)


        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']), actor_weights,
                             args['cascade_net'])

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars(), critic_weights,
                               args['cascade_net'])
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim),sigma=args['sigma'])


        train(sess, env, args, actor, critic, actor_noise)
        # if args['save_model']:
        #     saver.save(sess,'./results/tf_model/model')

        # if args['use_gym_monitor']:
        #     env.close()
    sess.close()

#def defalut hyper parameters,ORIGINALLY
ENV_NAME='Pendulum-v0'
ACTOR_LR=1E-3#1e-4
CRITIC_LR=1E-3#1e-3
GAMMA=0.99      #0.99
TAU=0.01 #1e-3
MINIBATCH_SIZE= 32#64
MAX_EPISODE_LEN=200#1000

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=ACTOR_LR)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=CRITIC_LR)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=GAMMA)
    parser.add_argument('--tau', help='soft target update parameter', default=TAU)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=20000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=MINIBATCH_SIZE)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default=ENV_NAME)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=12345)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=5000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=MAX_EPISODE_LEN)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')
    parser.add_argument('--save-model',help='save tensorflow model',action='store_true')
    parser.add_argument('--rewards-dir',help='directory for saving rewards',default='./results/rewards/')
    parser.add_argument('--sigma',help='sigma value for OU process',default=0.3,type=float)
    parser.add_argument('--rand-init',help='random initialize previous network weights',action='store_true')
    parser.add_argument('--cascade-net',help='construct two trainable cascade nets',action='store_true')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(save_model=False)
    parser.set_defaults(rand_init=False)
    parser.set_defaults(cascade_net=False)

    
    args = vars(parser.parse_args())
    main(args)
    
    
