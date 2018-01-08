from AirSimClient import *

from argparse import ArgumentParser

import numpy as np
from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer
import random
import csv
import math
#from math import pi as PI
np.set_printoptions(threshold=np.inf)
episode_count = 0
failcount = 0
landing = False
landing_count = 0
stuck_pt = np.array([0,0,0])
stuck_count = 0
#target_pt = np.array([-13,-19.5,-3])
target_pt = np.array([-8,-7,-1.678])
conti_done = 0
episode_step = 0
episode_limit = 500

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        #print np.zeros()
        self._states = np.zeros((size,) + sample_shape, dtype=np.uint8)
        print("!!!!!!!!!!!!!!!",self._states.shape)
        #time.sleep(5)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.

        Attributes:
            size (int): The minibatch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps
        self._rate = 0
        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            self._rate = self._step_size * step + self._start
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)

def huber_loss(y, y_hat, delta):
    """ Compute the Huber Loss as part of the model graph

    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2

    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold

    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(0.91, 0.1, 910000), fixpolicy=LinearEpsilonAnnealingExplorer(0.5, 0.1, 100000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=10000, train_interval=4, target_update_interval=10000,
                 monitor=True):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._fixpolicy = fixpolicy
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        print("input_shape:",input_shape)
        print("input_shape[1:]",input_shape[1:])
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # Action Value model (used by agent to interact with the environment)
        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                #Convolution2D((8, 8), 16, strides=4),
                #Convolution2D((4, 4), 32, strides=2),
                #Convolution2D((1, 1), 16, strides=1),
                Dense(25, init=he_uniform(scale=0.01)),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))
            ])
        self._action_value_net.update_signature(Tensor[input_shape])

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        # Function computing Q-values targets as part of the computation graph
        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            return element_select(
                terminals,
                rewards,
                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
            )

        # Define the loss, using Huber Loss (more robust to outliers)
        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions],
                   post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            # Compute the q_targets
            q_targets = compute_q_targets(post_states, rewards, terminals)

            # actions is a 1-hot encoding of the action done by the agent
            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)

            # Define training criterion as the Huber Loss function
            return huber_loss(q_targets, q_acted, 1.0)

        # Adam based SGD
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)
        #self._trainer.restore_from_checkpoint("models_heuristic_no_image\model")
    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)
        #if True:
        if self._fixpolicy.is_exploring(self._num_actions_taken):
            diff_x = state[3]-state[0]
            diff_y = state[4]-state[1]
            diff_z = state[5]-state[2]
            diff_arr = np.array([diff_x,diff_y,diff_z])
            direction = np.argmax(np.absolute(diff_arr))
            
            
            '''
            abs_x = math.fabs(diff_x)
            abs_y = math.fabs(diff_y)
            abs_z = math.fabs(diff_z)
            diff = [diff_x, diff_y, diff_z]
            abs_diff = [abs_x, abs_y, abs_z]
            print(diff, abs_diff)
            m = max(abs_diff)
            direction = diff.index(m)'''
            print(diff_arr)
            if diff_arr[direction]<0:
                fixaction = direction + 4
            else:
                fixaction = direction + 1
                
            self._num_actions_taken += 1
            return fixaction
        
        
        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                # Append batch axis with only one sample to evaluate
                env_with_history.reshape((1,) + env_with_history.shape)
            )

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state

        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """

        agent_step = self._num_actions_taken
        print("agent_step = ",agent_step)
        #time.sleep(1)
        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)

                self._trainer.train_minibatch(
                    self._trainer.loss_function.argument_map(
                        pre_states=pre_states,
                        actions=Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions),
                        post_states=post_states,
                        rewards=rewards,
                        terminals=terminals
                    )
                )

                # Update the Target Network if needed
                if (agent_step % self._target_update_interval) == 0:
                    self._target_net = self._action_value_net.clone(CloneMethod.freeze)
                    filename = "models_heuristic_no_image_less_exploration\model%d" % agent_step
                    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$filename=",filename)
                    self._trainer.save_checkpoint(filename)
                    #time.sleep(100)

    def _plot_metrics(self):
        global landing_count, episode_count
        """Plot current buffers accumulated values to visualize agent learning
        """
        f = open('log__heuristic_no_image_less_exploration2', 'a+')
        f.write('episode:' + str(episode_count) +': exploration rate= ' + str(self._explorer._rate) +' heuristic fix rate= ' + str(self._fixpolicy._rate) +'\n')
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)
            print('Mean Q per ep.', mean_q, self._num_actions_taken)
            f.write('Mean Q per ep. '+ str(mean_q) + ' ' + str(self._num_actions_taken) +'\n')

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)
            print('Mean Std Q per ep.', std_q, self._num_actions_taken)
            f.write('Mean Std Q per ep. ' +str(std_q) + ' ' + str(self._num_actions_taken)+ '\n')
            
        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)
        print('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)
        f.write('Sum rewards per ep. ' + str(sum(self._episode_rewards)) + ' ' + str(self._num_actions_taken) + '\n')
        if landing_count>0:
            f.write('****************Success landing**********' +str(landing_count)+'\n')
        landing_count = 0
        episode_count = 0
        
            
        f.write('\n')
def transform_input(responses):
    
    img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
    #print(img1d)
    #print("above is img1d :",img1d.shape)
    #time.sleep(5)
    
    #img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    '''print(img1d)
    print("above is img1d_2 :",img1d.size)
    time.sleep(5)'''
    
    img2d = img1d.reshape(responses[0].height, responses[0].width, 4)
    #img2d = img2d[:,:,0:2]
    #print(img2d)
    #print("above is img2d :",img2d.shape)
    #time.sleep(5)
    
    
    from PIL import Image
    image = Image.fromarray(img2d)
    
    '''
    image_t = Image.fromarray(img2d[:,:,0])
    image_test = image_t.convert("L")
    image_t.show()
    image_test.show()
    '''
    
    #image_t = image_t.convert("L")
    #im_final = np.array(image.resize((84, 84)).convert('L')) 
    im_final = np.array(image.resize((84, 84))) 
    #im_s = np.array(image_t.resize((84, 84)))
    #image_t.show()
    #print(im_final)
    #print("above is im_final :",im_final.shape)
    #print("above is im_single :",im_s.shape)
    #time.sleep(10)
    return im_final
    
    
    #responses[0].image_data_float=responses[0].image_data_float*255
    #print(responses[0].image_data_float)
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    print(img1d)
    print("above is img1d :",img1d.shape)
    time.sleep(5)
    
    #img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    '''print(img1d)
    print("above is img1d_2 :",img1d.size)
    time.sleep(5)'''
    
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    print(img2d)
    print("above is img2d :",img2d.shape)
    time.sleep(5)
    
    from PIL import Image
    image = Image.fromarray(img2d)
    #im_final = np.array(image.resize((84, 84)).convert('L')) 
    im_final = np.array(image.resize((84, 84))) 
    im_final = im_final*255
    print(im_final)
    print("above is im_final :",im_final.shape)
    time.sleep(1)
    return im_final


    
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    #img1d = np.array(responses[0].image_data_uint8, dtype=np.uint8)
    #print(img1d)
    #img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    #image.save("test.jpeg")
    #im_final = np.array(image.resize((84, 84)).convert('L')) 
    im_final = np.array(image.resize((84, 84)))
    return im_final

def interpret_action(action):
    scaling_factor = 0.25
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset

def compute_reward(current_state, quad_state, quad_vel, collision_info):
    global failcount, stuck_pt, stuck_count, landing, landing_count, target_pt, episode_count
    reward = 0
    landing = False
    quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
    
    '''
        for i in range(0, 83):
        for j in range(0, 83):
            if current_state[i,j,0]>95 and current_state[i,j,1]<100 and current_state[i,j,2]<100: #current_state[i,j,0]-current_state[i,j,1]-current_state[i,j,2] > 50 and
                reward = reward + 1 #current_state[i,j]*0.005
            elif current_state[i,j,0]>220 and current_state[i,j,1]<190 and current_state[i,j,2]<190:
                reward = reward + 1
    '''
    reward = -np.linalg.norm(quad_pt-target_pt)
    if collision_info.has_collided:
        if collision_info.object_id != 227:
            #reward = reward -200
        
            stuck_dis = np.linalg.norm(quad_pt-stuck_pt)
            thresh_dist = 15
            if stuck_dis< thresh_dist:
                reward = reward - 1000
                stuck_count += 1
                if stuck_count >30:
                    reward -= 200
            else:
                    stuck_count = 0
                    stuck_pt = quad_pt
        else:
            reward += 100 
        if collision_info.object_id == 132:
            reward += 1000 * (2*episode_limit - episode_count)/episode_limit #1000*(1+remain_episode persentage)
            landing = True
            landing_count += 1

    if reward <=-80:
        failcount = failcount+1
        if failcount > 15:
            reward -=100
    else:
        failcount = 0
    
    episode_count += 1
    return reward

    '''
    thresh_dist = 7
    beta = 1

    z = -10
    pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]

    quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
    print("point and velocity")
    print(quad_pt)
    #print(quad_vel)
    time.sleep(3)
    if collision_info.has_collided:
        reward = -100
    else:    
        dist = 10000000
        for i in range(0, len(pts)-1):
            dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

        #print(dist)
        if dist > thresh_dist:
            reward = -10
        else:
            reward_dist = (math.exp(-beta*dist) - 0.5) 
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            reward = reward_dist + reward_speed

    return reward'''

def isDone(reward):
    global landing_count, episode_count, episode_limit
    done = 0
    if  failcount > 15:
        done = 1
    if landing_count > 5:
        done = 1
    if episode_count > episode_limit:
        done = 1
    #episode_count = 0 # reset after write episode_count to file
    return done

'''initX = -.55265
initY = -31.9786
initZ = -19.0225'''
initX = 0
initY = 0
initZ = -6

# connect to the AirSim simulator 
client = MultirotorClient(41451)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoff()
print('start init')
client.moveToPosition(initX, initY, initZ, 1)
print('end init')
quad_state = client.getPosition()
quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
initX=quad_state.x_val
initY=quad_state.y_val
initZ=quad_state.z_val

print("point: ")
print(quad_pt)



#client.moveByAngle(0,0,0,1,8)
#print("I rotateeeeee")
'''
client.moveByVelocity(3, 5, -1, 5)
time.sleep(7)
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoff()'''
'''time.sleep(7)
quad_state = client.getPosition()
quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
print("point after move: ")
print(quad_pt)
time.sleep(3)

#client.moveToPosition(initX-quad_state.x_val, initY-quad_state.y_val, initZ, 5)
client.moveToPosition(0, 0, initZ, 5)
time.sleep(7)
quad_state = client.getPosition()
quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
print("point after move back: ")
print(quad_pt)
time.sleep(30)'''
# Make RL agent
NumBufferFrames = 4
SizeRows = 9
SizeCols = 1
NumActions = 7
#agent = DeepQAgent((NumBufferFrames, SizeRows), NumActions, train_after=1000, target_update_interval=1000, monitor=True)
agent = DeepQAgent((NumBufferFrames, SizeRows), NumActions,
        train_after=1000, target_update_interval=1000, monitor=True,
        explorer=LinearEpsilonAnnealingExplorer(1, 0.05, 10000), fixpolicy=LinearEpsilonAnnealingExplorer(0.7, 0.1, 100000),
        learning_rate=0.0025)
# Train
epoch = 100
current_step = 0
max_steps = epoch * 250000

'''responses = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
AirSimClientBase.write_file(os.path.normpath('c:/temp/py_drone_test3' + '.png'), responses[0].image_data_uint8)
print (responses[0].image_data_uint8)'''

'''
responses = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, True, False)])
#print (responses[0].image_data_float)
#time.sleep(3)
#AirSimClientBase.write_file(os.path.normpath('c:/temp/py_drone_test2' + '.png'), responses[0].image_data_float)
current_state = transform_input(responses)
#print (current_state)
'''

#GPS = client.getGpsLocation
#print("GPS:", GPS)

''' #angle initial between -145~155   angle range -180~180 (crash whe over 180, so don't move to 180 too fast)
client.moveByAngle(0,0,0,-145*PI/180,7)
time.sleep(7)
Orient = client.getOrientation()
print("-------------------------Orient:", Orient.x_val, Orient.y_val, Orient.z_val, Orient.w_val)
time.sleep(1)

client.moveByAngle(0,0,0,-155*PI/180,7)
time.sleep(7)
Orient = client.getOrientation()
print("-------------------------Orient:", Orient.x_val, Orient.y_val, Orient.z_val, Orient.w_val)
time.sleep(1)
'''


#client.moveByVelocity(-4, -6, 1, 7) #move to block
'''
client.moveToPosition(-8, -7, -5,3)
time.sleep(5)
client.moveByVelocity(0,0,1,5)
time.sleep(5)'''


#responses = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
#current_state = transform_input(responses)
quad_state = client.getPosition()
quad_vel = client.getVelocity()
quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
current_state = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val, target_pt[0], target_pt[1], target_pt[2], quad_vel.x_val, quad_vel.y_val, quad_vel.z_val)))

collision_info = client.getCollisionInfo()
print(collision_info.has_collided)
print(collision_info.object_id)
print(collision_info.object_name)
print(collision_info.normal)
print("&&&&&&&&&&&&&&   quad point=",quad_pt)
time.sleep(5)
while True:
    action = agent.act(current_state)
    quad_offset = interpret_action(action)
    quad_vel = client.getVelocity()
    
    client.moveByVelocity(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 2) #last arg is duration
    time.sleep(0.5)
 
    quad_state = client.getPosition()
    quad_vel = client.getVelocity()
    collision_info = client.getCollisionInfo()
    
    reward = compute_reward(current_state, quad_state, quad_vel, collision_info)
    done = isDone(reward)
    print('Action, Reward, Done:', action, reward, done)
    
    agent.observe(current_state, action, reward, done)
    agent.train()
    #time.sleep(2.5)
    if done:
        print("I will reset to init: ")
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoff()
        '''
        quad_state = client.getPosition()
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        print("now point: ")
        print(quad_pt)
        #client.moveToPosition(initX-quad_state.x_val, initY-quad_state.y_val, initZ-quad_state.z_val, 5)
        client.moveToPosition(0, 0, -30, 5)
        client.moveToPosition(initX, initY, initZ, 5)
        quad_state = client.getPosition()
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        print("point: ")
        print(quad_pt)
        #client.moveByVelocity(1, -0.67, -0.8, 5)
        time.sleep(5)
        current_step +=1
        conti_done +=1
        if conti_done>=2:
            angle = random.uniform(0, 3.6)
            client.moveByAngle(0,0,0,angle,4)
            print("I rotateeeeee", angle)
            time.sleep(5)
            #client.moveByAngle()
        '''
    else:
        conti_done = 0
    
    #responses = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
    
    #current_state = transform_input(responses)
    current_state = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val, target_pt[0], target_pt[1], target_pt[2], quad_vel.x_val, quad_vel.y_val, quad_vel.z_val)))
    #print (current_state)

    
