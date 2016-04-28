# Imports.
import numpy as np
import numpy.random as npr
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.X_train = np.zeros(shape=(1892, 3), dtype=int)
        self.Y_train = np.empty(shape=1892, dtype=int)

        with open('training3d_2nd.txt', 'r') as f:
            count = 0
            for line in f:
                x = line.split(';')[0]
                x = x.replace(" ", "")
                x = x.replace("[", "")
                x = x.replace("]", "")
                params = x.split(',')
                new = []
                for param in params:
                    new.append(int(param))

                self.X_train[count] = new

                y = line.split(';')[1]
                self.Y_train[count] = int(y)

                count += 1

        self.clf = svm.SVC(kernel='linear')
        self.clf.fit(self.X_train, self.Y_train)

        # self.clf = RandomForestClassifier()
        # self.clf.fit(self.X_train, self.Y_train)

        # print "count_lines = %d" % count
        # print "Y_train size = %d" % len(self.Y_train)
        # print "X_train size = %d" % len(self.X_train)


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # new_action = npr.rand() < 0.1
        # new_state  = state

        # self.last_action = new_action
        # self.last_state  = new_state

        # return self.last_action

        delta_x = state['tree']['dist']
        delta_y = state['monkey']['bot'] - state['tree']['bot']
        V = state['monkey']['vel']
        # state_info = [delta_x, delta_y, V, pow(delta_x, 2), pow(delta_y, 2), pow(V, 2), 
        #         pow(delta_x, 3), pow(delta_y, 3), pow(V, 3)]
        state_info = [delta_x, delta_y, V,]

        state_info = np.asarray(state_info)
        state_info = state_info.reshape(1, -1)

        act = self.clf.predict(state_info)
        print act

        return act

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


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
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


