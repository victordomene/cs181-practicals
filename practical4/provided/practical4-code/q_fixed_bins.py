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
GAMMA = 0.5 # discount factor
EPS = 1.0 # e-greedy start parameter
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

        # Q Matrix
        self.Q = np.zeros((2, 2, 2, 3))
        self.k = np.zeros((2, 2, 2, 3))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def _transform(self, t, d):
	# velocity
        if t == 'v':
		if d < -10:
			return 0
		elif d < 0: 
			return 1
		else:
			return 2
	# height
        elif t == 'h':
		if d < 0:
			return 0
		else:
			return 1
	# width
        elif t == 'w':
		if d < 185:
			return 0
		else:
			return 1
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

	d_tree = self._transform('w', state['tree']['dist'])
        horiz_delta = self._transform('h', (state['tree']['top'] + state['tree']['bot'])/2 - (state['monkey']['bot'] + state['monkey']['top'])/2)
        vel = self._transform('v', state['monkey']['vel'])

        # D: pixels to next tree
        # T: height of bottom of tree
        # M: height of the monkey
	# print d_tree, horiz_delta, vel

        new_action = np.argmax(self.Q[:, d_tree, horiz_delta, vel])
        self.k[new_action, d_tree, horiz_delta, vel] += 1
        eps = EPS/(self.k[new_action, d_tree, horiz_delta, vel])

        # for your first step, do something random.
        if self.last_action == None:
            new_action = self.random_move()
	else:
		# Update Q function
		d_tree2, horiz_delta2, vel2 = self.last_state
		max_q = np.max(self.Q[:, d_tree, horiz_delta, vel])
		old_val = self.Q[self.last_action, d_tree2, horiz_delta2, vel2]
		alpha = 1 / (self.k[self.last_action, d_tree, horiz_delta, vel] + 1)

		new_val = old_val + alpha * (self.last_reward + GAMMA * max_q - old_val)
		self.Q[self.last_action, d_tree2, horiz_delta2, vel2] = new_val
		if (npr.rand() < eps):
			new_action = self.random_move()

        self.last_action = new_action
        self.last_state = d_tree, horiz_delta, vel 
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        r = self.get_reward(reward)
	self.last_reward = r

	if r < 0:
		print "Exploration Rate: {}".format(float(np.count_nonzero(self.k)) / self.k.size) 


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
	print "Score: {}".format(swing.score)
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
  run_games(agent, hist, 1000, 1)

  # Save history.
  np.save('hist',np.array(hist))


