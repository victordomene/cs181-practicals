import numpy as np

class ManualFactorizer:
	def __init__(self):
		return

	def factorize(self, X, artists, users, W = None, H = None):
		""" 
		Returns two NumPy arrays containing the factorization of X. This 
		EXCLUSIVELY factorizes 1 factor. Our experiments have shown that there
		is not a lot of data to be learned here, and therefore using a model
		that has only a multiplier per user and another one per-artist seems
		to be reasonable.
		
		:param X: a NumPy array to be factorized. 
		:return W, H: two NumPy arrays (the factorization).
		"""

		self.X = X

		if W is None or H is None:
			self.W = np.random.normal(1.0, 0.2, X.shape[0])
			print self.W.shape
			self.H = np.random.normal(1.0, 0.2, X.shape[1])
			print self.H.shape
		else:
			self.W = W
			self.H = H

		# number of artists and users; this is probably not necessary
		# since it would be the value in X.shape, but ok
		self.num_artists = len(artists)
		self.num_users = len(users)

		# learning parameters
		self.gamma = 0.001
		
		self.bias_for_user = []
		for user in xrange(self.num_users):
			median = np.median(X[user][X[user] != 0])
			self.bias_for_user.append(median)

		self.W = np.array(self.bias_for_user)
		self.H = np.array([1.0 for _ in xrange(X.shape[1])])

		max_iter = 10

		for i in xrange(max_iter):
			print "Doing {}-th iteration...".format(i)
			self.optimizeUsers()
			# self.optimizeArtists()
			# self.optimizeAll()

		print "# Getting reconstruction error..."
		print self.reconstructionError()

		return self.W, self.H

	def reconstructionError(self):
		error = 0.0
		count = 0

		for user, plays_for_user in enumerate(self.X):
			for artist, plays in enumerate(plays_for_user):
				if plays > 0.0:
					count += 1
					error += abs(plays - self.W[user] * self.H[artist])

		return error / count

	def optimizeArtists(self):
		"""
		Fixes the users, and optimize the multipliers for the artists.
		Returns the newly optimized W matrix.
		
		:param X:
		:param W:
		:param H:

		:return W:
		"""
		
		# for each artist, we do a gradient descent as specified
		# in the Vollinsky paper, fixing the users.

		num_iter = 3

		for _ in xrange(num_iter):
			for artist in xrange(self.num_artists):
				if artist % 100 == 0:
					print artist

				X_artist = self.X[:, artist]
				# print X_artist
				args = np.argwhere(X_artist > 0.0)


				for user, plays_for_user in zip(args, X_artist[args]):
					# update with gradients
					err = plays_for_user - self.W[user] * self.H[artist] 
					# print err
					# quit()

					self.H[artist] += (self.gamma * err) * self.W[user] 


		print "median of H: {}".format(np.median(self.H))
		print "min of H: {}".format(np.amin(self.H))
		print "max of H: {}".format(np.amax(self.H))
		return self.H

	def optimizeUsers(self):
		"""
		Fixes the artists, and optimize the multipliers for the users.
		Returns the newly optimized H matrix.
		
		:param X:
		:param W:
		:param H:

		:return W:
		"""
		
		# for each artist, we do a gradient descent as specified
		# in the Vollinsky paper, fixing the users.
		
		num_iter = 3

		for _ in xrange(num_iter):
			for user in xrange(self.num_users):
				if user % 10000 == 0:
					print user

				X_user = self.X[user]
				# print X_artist
				args = np.argwhere(X_user > 0.0)

				for artist, plays_for_artist in zip(args, X_user[args]):
					# update with gradients
					err = self.gamma * (plays_for_artist - np.dot(self.W[user], self.H[artist]))

					if err > 0.0:
						self.W[user] += self.gamma * self.H[artist]
						self.H[artist] += self.gamma * self.W[user]
					else:
						self.W[user] -= self.gamma * self.H[artist]
						self.H[artist] -= self.gamma * self.W[user]
		
		print "median of W: {}".format(np.median(self.W))
		print "min of W: {}".format(np.amin(self.W))
		print "max of W: {}".format(np.amax(self.W))
		print "median of H: {}".format(np.median(self.H))
		print "min of H: {}".format(np.amin(self.H))
		print "max of H: {}".format(np.amax(self.H))
		return self.W

def main():
	est = ManualFactorizer()

	X = np.array([[500.,400.,300.],[700.,504.,300.],[600.,400.,0.0]])
	
	median_for_users = []
	for user in X:
		median_for_users.append(np.median(user))
	
	artists = np.array([1,1,1])
	users = np.array([2,2,2])

	W2, H2 = est.factorize(X, artists, users)
	
	print W2
	print H2
	
	print W2[2] * (H2[2]) + median_for_users[2]
	print W2[0] * H2[1] + median_for_users[0]

