#### Here, we try:
# horiztonal distance
# vertical delta
# velocity

# for the Q function.
# with binnings, of course.

# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
MAX_VELOCITY = 60

# Factors for experimentation
BIN_V = 1
BIN_DT = 1 # Dist tree
BIN_DB = 1 # Dist bottom
GAMMA = 0.1 # discount factor
EPS = 0.01 # e-greedy start parameter
EPS_EXPONENT = 1.3
ALPHA = 0.2

class Learner(object):
    '''
    This agent uses Q-learning.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.epoch = 0
        self.iters = 0 # number of iterations in the epoch

        self.eps = EPS
        self.v_history = []
        '''
        Q matrix:
        ndarray of dimensions A x D x T x M

        A: action to perform
        D: pixels to next tree
        T: height of bottom of tree
        M: height of the monkey

        Before updating the Q function, the wrapper function _transform
        needs to be run on the data.
        '''

        # TODO: What is the best way to initialize these functions?

        # Q Matrix
        # self.Q = np.random.choice([-0.1,-0.08,-0.06,-0.01,0,0.1,0.08,0.06,0.01], size=(2, self._transform('w', SCREEN_WIDTH), self._transform('h', SCREEN_HEIGHT), self._transform('h', SCREEN_HEIGHT)))

        self.Q = np.zeros((2, self._transform('w', SCREEN_WIDTH), self._transform('h', SCREEN_HEIGHT), self._transform('v', MAX_VELOCITY) * 2))

        self.k = np.zeros((2, self._transform('w', SCREEN_WIDTH), self._transform('h', SCREEN_HEIGHT),  self._transform('v', MAX_VELOCITY) * 2))

        # Number of times actions have been taken
        # Used for epsilon greedy
        # self.k = np.zeros((2, self._transform('w', SCREEN_WIDTH), self._transform('h', SCREEN_HEIGHT), self._transform('h', SCREEN_HEIGHT)))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    # a collection of wrappers over velocity, distances, etc.
    # that we can later use for binning and preprocessing.

    def _v(self, v):
        return v

    def _dist_tree(self, d):
        return d

    def _dist_bot(self, d):
        return d

    def _transform(self, t, d):
        if t == 'v': # 'velocity' :
            # res = (d + 15) / 5
            # if np.abs(res) > 6:
            #     res = np.sign(res) * 6
            res = d / 10
            if np.abs(res) > 4:
                res = np.sign(res) * 4
            res += 4
            return res
        elif t == 'h': # height
            return (d + SCREEN_HEIGHT) / 20
        elif t == 'w': # width
            return d / 50
        return d

    def get_reward(self, r):
        return r

    def random_move(self):
        return npr.rand() < 0.5

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        self.iters += 1
        # if self.iters == 10:
        # print self.Q
        # self.eps /= EPS_CHANGE
        # print state
        # extract information
        d_tree = self._transform('w', state['tree']['dist'])
        if d_tree < 0:
            d_tree = 0

        horiz_delta = self._transform('h', state['tree']['bot'] - state['monkey']['bot'])
        vel = self._transform('v', state['monkey']['vel'])

        # D: pixels to next tree
        # T: height of bottom of tree
        # M: height of the monkey

        new_action = np.argmax(self.Q[:, d_tree, horiz_delta, vel])
        self.k[new_action, d_tree, horiz_delta, vel] += 1
        eps = EPS/(self.k[new_action, d_tree, horiz_delta, vel] ** EPS_EXPONENT)

        if self.last_action == None or npr.rand() < eps:
            # for your first step, do something random.
            new_action = self.random_move()

        self.last_action = new_action
        self.last_state = (d_tree, horiz_delta, vel)
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        # transform reward
        r = self.get_reward(reward)
        # print r

        # Update Q function
        d_tree, horiz_delta, vel = self.last_state
        #print self.Q[:, d_tree, b_tree, b_monkey, vel]
        max_q = np.max(self.Q[:, d_tree, horiz_delta, vel])
        old_val = self.Q[self.last_action, d_tree, horiz_delta, vel]
        alpha = 1 / (self.k[self.last_action, d_tree, horiz_delta, vel] + 1)
        new_val = old_val + alpha * (r + GAMMA * max_q - old_val)
        self.Q[self.last_action, d_tree, horiz_delta, vel] = new_val

        # if you dye...
        if reward < 0:
            self.eps = EPS

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

  # Select agent.
  agent = Learner()

  # Empty list to save history.
  hist = []

  # Run games.
  run_games(agent, hist, 10000000, 1)

  # Save history.
  np.save('hist',np.array(hist))


