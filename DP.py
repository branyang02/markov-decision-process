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

		policy = np.zeros(self.nStates)
		V = initialV
		iterId = 0 
		epsilon = tolerance

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
		print("Number of Iterations of Value Iteration: ", iterId)

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

		policy = initialPolicy
		V = np.zeros(self.nStates)
		iterId = 0
		
		while iterId < nIterations:
			iterId += 1
			# Policy Evaluation Linear Systems of Equations
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)
			# Policy Improvement
			policy, policy_stable = self.extractPolicy(V)
			if policy_stable is True:
				break
		print("Number of iterations of Policy Iteration: ", iterId)

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

		# construct the Transition Matrix following current policy
		Transition_Matrix = []
		for i in range(self.nStates):
			Transition_Matrix.append(self.T[int(policy[i])][i].tolist())
		Transition_Matrix = np.array(Transition_Matrix)
		# Calculate Value Function using system of linear equations
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
		V = initialV
		policy = initialPolicy
		iterId = 0
		epsilon = tolerance

		while iterId < nIterations:
			iterId += 1
			# Partial Policy Evaluation using iteratively method
			policy_iterId = 0
			while policy_iterId < nPolicyEvalIterations:
				policy_iterId += 1
				delta = 0
				for current_state in range(self.nStates):
					v = V[current_state]
					action = int(policy[current_state])
					V[current_state] = sum(self.T[action, current_state] * (self.R[action][current_state] + (self.discount * V)))
					delta = max(delta, abs(v - V[current_state]))
				if delta < tolerance:
					break
			# Policy Improvement
			policy, policy_stable = self.extractPolicy(V)
			if policy_stable is True:
				break

		return [policy, V, iterId, epsilon]

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