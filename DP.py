from MDP import build_mazeMDP, print_policy
import numpy as np

class DynamicProgramming:
	def __init__(self, MDP):
		self.R = MDP.R
		self.T = MDP.T
		self.discount = MDP.discount
		self.nStates = MDP.nStates
		self.nActions = MDP.nActions
		self.up = 0
		self.down = 1
		self.left = 2
		self.right = 3


	def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
		'''Value iteration procedure
		V <-- max_a R^a + gamma T^a V

		Inputs:
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		policy = np.zeros(self.nStates) # number corresponds to each action at each state
		V = initialV
		iterId = 0 
		epsilon = tolerance


		while iterId < nIterations:
			tempV = np.zeros(self.nStates)
			for current_state in range(self.nStates):
				up_value = sum(self.T[self.up, current_state] * (self.R[self.up][current_state] + self.discount * V))
				down_value = sum(self.T[self.down, current_state] * (self.R[self.down][current_state] + self.discount * V))
				left_value = sum(self.T[self.left, current_state] * (self.R[self.left][current_state] + self.discount * V))
				right_value = sum(self.T[self.right, current_state] * (self.R[self.right][current_state] + self.discount * V))

				action_list = [up_value, down_value, left_value, right_value]
				tempV[current_state] = max(action_list)
				policy[current_state] = action_list.index(max(action_list))
			if False not in np.isclose(tempV, V, rtol=epsilon):
				break
			iterId += 1
			V = tempV

		return [policy, V, iterId, epsilon]

	def policyIteration_v1(self, initialPolicy, nIterations=np.inf, tolerance=0.01):
		'''Policy iteration procedure: alternate between policy
		evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
		improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		nIterations -- limit on # of iterations: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		policy = initialPolicy
		V = np.zeros(self.nStates)
		iterId = 0

		return [policy, V, iterId]


	def extractPolicy(self, V):
		'''Procedure to extract a policy from a value function
		pi <-- argmax_a R^a + gamma T^a V

		Inputs:
		V -- Value function: array of |S| entries

		Output:
		policy -- Policy: array of |S| entries'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		policy = np.zeros(self.nStates)

		return policy


	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi

		Input:
		policy -- Policy: array of |S| entries

		Ouput:
		V -- Value function: array of |S| entries'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		V = np.zeros(self.nStates)

		return V

	def policyIteration_v2(self, initialPolicy, initialV, nPolicyEvalIterations=5, nIterations=np.inf, tolerance=0.01):
		'''Modified policy iteration procedure: alternate between
		partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
		and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nPolicyEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
		nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		policy = np.zeros(self.nStates)
		V = np.zeros(self.nStates)
		iterId = 0
		epsilon = 0

		return [policy, V, iterId, epsilon]

	def evaluatePolicy_IterativeUpdate(self, policy, initialV, nIterations=np.inf):
		'''Partial policy evaluation:
		Repeat V^pi <-- R^pi + gamma T^pi V^pi

		Inputs:
		policy -- Policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)

		Outputs:
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		V = np.zeros(self.nStates)
		iterId = 0
		epsilon = 0

		return [V, iterId, epsilon]


if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)
	# Test policy iteration v1
	[policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, dtype=int))
	print_policy(policy)
	# Test policy iteration v2
	[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)