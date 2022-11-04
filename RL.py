import sys
import matplotlib.pyplot as plt

from MDP import build_mazeMDP, print_policy
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline

class ReinforcementLearning:
	def __init__(self, mdp, sampleReward):
		"""
		Constructor for the RL class

		:param mdp: Markov decision process (T, R, discount)
		:param sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian). This function takes one argument:
		the mean of the distribution and returns a sample from the distribution.
		"""

		self.mdp = mdp
		self.sampleReward = sampleReward

	def sampleRewardAndNextState(self, state, action):
		'''Procedure to sample a reward and the next state
		reward ~ Pr(r)
		nextState ~ Pr(s'|s,a)

		Inputs:
		state -- current state
		action -- action to be executed

		Outputs:
		reward -- sampled reward
		nextState -- sampled next state
		'''

		reward = self.sampleReward(self.mdp.R[action,state])
		cumProb = np.cumsum(self.mdp.T[action,state,:])
		nextState = np.where(cumProb >= np.random.rand(1))[0][0]
		return [reward, nextState]

	def OffPolicyTD(self, nEpisodes, epsilon=0.0, alpha=0.1):
		Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		policy = np.zeros(self.mdp.nStates,int)
		cumulative_reward = []
		for i_episode in range(1, nEpisodes+1):
			if i_episode % 1000 == 0:
				print("\rEpisode {}/{}.".format(i_episode, nEpisodes), end="")
				sys.stdout.flush()
			
			sum_rewards = 0
			state = np.random.randint(16)
			while True:
				policy[state] = np.argmax(Q[:, state])
				# epsilon greedy
				explore = np.random.binomial(1, epsilon)
				if explore == 1:
					action = np.random.randint(4)
				else:
					action = policy[state]
				
				reward, next_state = self.sampleRewardAndNextState(state, action)
				sum_rewards += reward
				best_next_action = np.argmax(Q[:, next_state])
				Q[action][state] += alpha * (reward + (self.mdp.discount * Q[best_next_action][next_state]) - Q[action][state])
				state = next_state
				if next_state == 16:  # if the terminal state is reached, break
					break
			cumulative_reward.append(sum_rewards)
		return [Q,policy, cumulative_reward]

	def OffPolicyMC(self, nEpisodes, epsilon=0.1):
		total_return = []
		Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		policy = np.zeros(self.mdp.nStates,int)  		
		C = np.zeros_like(Q)
		nSteps = 100_000_000
		for i_episode in range(1, nEpisodes+1):
			if i_episode % 1000 == 0:
				print("\rEpisode {}/{}.".format(i_episode, nEpisodes), end="")
				sys.stdout.flush()
			# Generate an episode using b: S0, A0, R1,...,ST-1, AT-1, RT
			episode = []
			# make starting state random
			state = np.random.randint(16)
			
			for t in range(nSteps):
				action = np.random.randint(4)
				reward, next_state = self.sampleRewardAndNextState(state, action)
				episode.append((state, action, reward))
				if next_state == 16:  # if the terminal state is reached, break
					break
				state = next_state
			
			G = 0.0
			W = 1.0
			
			for t in range(len(episode))[::-1]:
				state, action, reward = episode[t]
				G = self.mdp.discount * G + reward
				C[action][state] += W
				Q[action][state] += (W / C[action][state]) * (G - Q[action][state])
				policy[state] = np.argmax(Q[:, state])				
				if action != policy[state]:
					break
				W = W / (1/self.mdp.nActions)
			
			total_return.append(G)
		
		return [Q, policy, total_return]

if __name__ == '__main__':
	mdp = build_mazeMDP()
	rl = ReinforcementLearning(mdp, np.random.normal)  # reward is Normally Distributed with varience 1.0

	# Test Q-learning
	# TD_num_episode = 10_000
	# total_reward = [0] * TD_num_episode
	# for i in range(10):
	# 	[Q, policy, cumulative_reward] = rl.OffPolicyTD(nEpisodes=TD_num_episode, epsilon=0.1)
	# 	print_policy(policy)
	# 	for i in range(len(cumulative_reward)):
	# 		total_reward[i] += cumulative_reward[i]

	# for i in range(len(total_reward)):
	# 	total_reward[i] = total_reward[i]/10

	# y = np.array(total_reward)
	# x = np.array(list(range(1, TD_num_episode+1)))

	# xnew = np.linspace(x.min(), x.max(), 200) 

	# #define spline with degree k=7
	# spl = make_interp_spline(x, y, k=7)
	# y_smooth = spl(xnew)

	# #create smooth line chart 
	# plt.plot(xnew, y_smooth)
	# plt.show()


	# Test Off-Policy MC
	MC_num_episode = 30_000
	total_reward = [0] * MC_num_episode
	for i in range(10):
		[Q, policy, total_return] = rl.OffPolicyMC(nEpisodes=MC_num_episode, epsilon=0.1)
		print_policy(policy)
		for i in range(len(total_return)):
			total_reward[i] += total_return[i]
	
	for i in range(len(total_reward)):
		total_reward[i] = total_reward[i]/10


	y = np.array(total_reward)
	x = np.array(list(range(1, MC_num_episode+1)))

	xnew = np.linspace(x.min(), x.max(), 200) 

	#define spline with degree k=7
	spl = make_interp_spline(x, y, k=7)
	y_smooth = spl(xnew)

	#create smooth line chart 
	plt.plot(xnew, y_smooth)
	plt.show()