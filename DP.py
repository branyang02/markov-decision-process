from calendar import c
from MDP import build_mazeMDP, print_policy
import numpy as np

class DynamicProgramming:
	def __init__(self, MDP):
		self.R = MDP.R
		self.T = MDP.T
		self.discount = MDP.discount
		self.nStates = MDP.nStates
		self.nActions = MDP.nActions


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

		# self.up = 0
		# self.down = 1
		# self.left = 2
		# self.right = 3

		# while iterId < nIterations:
		# 	tempV = np.zeros(self.nStates)
		# 	for current_state in range(self.nStates):
		# 		up_value = sum(self.T[self.up, current_state] * (self.R[self.up][current_state] + self.discount * V))
		# 		down_value = sum(self.T[self.down, current_state] * (self.R[self.down][current_state] + self.discount * V))
		# 		left_value = sum(self.T[self.left, current_state] * (self.R[self.left][current_state] + self.discount * V))
		# 		right_value = sum(self.T[self.right, current_state] * (self.R[self.right][current_state] + self.discount * V))

		# 		action_list = [up_value, down_value, left_value, right_value]
		# 		tempV[current_state] = max(action_list)
		# 		policy[current_state] = action_list.index(max(action_list))
		# 	if False not in np.isclose(tempV, V, rtol=epsilon):
		# 		break
		# 	iterId += 1
		# 	V = tempV
		# print(V)

		while iterId < nIterations:
			iterId += 1
			delta = 0
			for current_state in range(self.nStates):
				v = V[current_state]
				max_value = -np.inf
				for action in range(self.nActions):
					value = sum(self.T[action, current_state] * (self.R[action][current_state] + (self.discount * V)))
					if value > max_value:
						max_value = value
				V[current_state] = max_value
				delta = max(delta, abs(v - V[current_state]))
			if delta < epsilon:
				break
		# output a deterministic policy
		for current_state in range(self.nStates):
			max_value = -np.inf
			selected_action = int(policy[current_state])
			for action in range(self.nActions):
				value = sum(self.T[action, current_state] * (self.R[action][current_state] + (self.discount * V)))
				if value > max_value:
					max_value = value
					selected_action = action
			policy[current_state] = selected_action
		print(iterId)
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
		
		while iterId < nIterations:
			iterId += 1
			# Policy Evaluation Linear Systems of Equations
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)
	
			# Policy Evaluation
			# while True:
			# 	delta = 0
			# 	for current_state in range(self.nStates):
			# 		v = V[current_state]
			# 		action = int(policy[current_state])
			# 		V[current_state] = sum(self.T[action, current_state] * (self.R[action][current_state] + (self.discount * V)))
			# 		delta = max(delta, abs(v - V[current_state]))
			# 	if delta < tolerance:
			# 		break
			
			# Policy Improvement
			policy, policy_stable = self.extractPolicy(V)
			if policy_stable is True:
				break
		print(iterId)
		return [policy, V, iterId]


	def extractPolicy(self, V):
		'''Procedure to extract a policy from a value function
		pi <-- argmax_a R^a + gamma T^a V

		Inputs:
		V -- Value function: array of |S| entries

		Output:
		policy -- Policy: array of |S| entries'''

		policy_stable = True
		for current_state in range(self.nStates):
			old_action = int(policy[current_state])
			# pick action that maximizes value
			max_value = -np.inf
			selected_action = old_action
			for action in range(self.nActions):
				value = sum(self.T[action, current_state] * (self.R[action][current_state] + (self.discount * V)))
				if value > max_value:
					max_value = value
					selected_action = action
			policy[current_state] = selected_action
			if old_action != policy[current_state]:
					policy_stable = False

		return policy, policy_stable


	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi

		Input:
		policy -- Policy: array of |S| entries

		Ouput:
		V -- Value function: array of |S| entries'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		# V = np.zeros(self.nStates)
		

		# construct the Transition Matrix following current policy
		Transition_Matrix = []
		for i in range(self.nStates):
			Transition_Matrix.append(self.T[int(policy[i])][i].tolist())
		Transition_Matrix = np.array(Transition_Matrix)

		# Calculate Value Function
		V = np.dot(np.linalg.inv((np.identity(self.nStates) - (self.discount * Transition_Matrix))), self.R[0])
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