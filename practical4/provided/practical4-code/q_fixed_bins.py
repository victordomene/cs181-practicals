#### Here, we try:
# horiztonal distance
# vertical delta
# velocity

# for the Q function.
# with binnings, of course. This uses a fixed (small) number of bins.

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
	self.gravity = None

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
        self.Q = np.zeros((2, 2, 2, 3, 2))
        self.k = np.zeros((2, 2, 2, 3, 2))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
	self.gravity = None

    # Defines the bins. Values are hard-coded.
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

	# Find all the features accordingly
	d_tree = self._transform('w', state['tree']['dist'])
        vert_delta = self._transform('h', (state['tree']['top'] + state['tree']['bot'])/2 - (state['monkey']['bot'] + state['monkey']['top'])/2)
        vel = self._transform('v', state['monkey']['vel'])
	pos = state['monkey']['bot']

        # for your first step, do not jump
        if self.last_action == None:
		new_action = 0
	else:
		# Update Q function
		old_dtree, old_vert_delta, old_vel, lastpos = self.last_state
		
		# If we do not have gravity yet, do not jump until we do.
		if self.gravity is None:
			if lastpos - pos == 4:
				self.gravity = 0
			elif lastpos - pos == 1:
				self.gravity = 1
			else:
				self.last_action = 0
				self.last_state = d_tree, vert_delta, vel, state['monkey']['bot']
				
				# Choose not to jump!
				return 0
		
		# Now we have the gravity. Find the best action in the Q table
		new_action = np.argmax(self.Q[:, d_tree, vert_delta, vel, self.gravity])

		# Update our k (used for alpha/epsilon updates)
		self.k[new_action, d_tree, vert_delta, vel, self.gravity] += 1

		# Find the appropriate epsilon and alpha
		eps = EPS/(self.k[new_action, d_tree, vert_delta, vel, self.gravity])
		alpha = 1 / (self.k[self.last_action, d_tree, vert_delta, vel, self.gravity] + 1)

		# Get the maximum Q for this state
		max_q = np.max(self.Q[:, d_tree, vert_delta, vel, self.gravity])

		# Run the Q update rule
		old_val = self.Q[self.last_action, old_dtree, old_vert_delta, old_vel, self.gravity]
		new_val = old_val + alpha * (self.last_reward + GAMMA * max_q - old_val)
		self.Q[self.last_action, old_dtree, old_vert_delta, old_vel, self.gravity] = new_val

		# Finally, use epsilon greedy!
		if (npr.rand() < eps):
			new_action = self.random_move()

	# Record the last action and last state
        self.last_action = new_action
        self.last_state = d_tree, vert_delta, vel, state['monkey']['bot'] 

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        r = self.get_reward(reward)
	self.last_reward = r

	# Print the exploration rate
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


