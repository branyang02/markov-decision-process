from calendar import c
from cmath import isclose
from dis import dis
from operator import le
from tkinter import NS
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# Transition function: |A| x |S| x |S'| array
T = np.zeros([4, 17, 17])
a = 0.8;  # intended move
b = 0.1;  # lateral move

# up (a = 0)

T[0, 0, 0] = a + b;
T[0, 0, 1] = b;

T[0, 1, 0] = b;
T[0, 1, 1] = a;
T[0, 1, 2] = b;

T[0, 2, 1] = b;
T[0, 2, 2] = a;
T[0, 2, 3] = b;

T[0, 3, 2] = b;
T[0, 3, 3] = a + b;

T[0, 4, 4] = b;
T[0, 4, 0] = a;
T[0, 4, 5] = b;

T[0, 5, 4] = b;
T[0, 5, 1] = a;
T[0, 5, 6] = b;

T[0, 6, 5] = b;
T[0, 6, 2] = a;
T[0, 6, 7] = b;

T[0, 7, 6] = b;
T[0, 7, 3] = a;
T[0, 7, 7] = b;

T[0, 8, 8] = b;
T[0, 8, 4] = a;
T[0, 8, 9] = b;

T[0, 9, 8] = b;
T[0, 9, 5] = a;
T[0, 9, 10] = b;

T[0, 10, 9] = b;
T[0, 10, 6] = a;
T[0, 10, 11] = b;

T[0, 11, 10] = b;
T[0, 11, 7] = a;
T[0, 11, 11] = b;

T[0, 12, 12] = b;
T[0, 12, 8] = a;
T[0, 12, 13] = b;

T[0, 13, 12] = b;
T[0, 13, 9] = a;
T[0, 13, 14] = b;

T[0, 14, 13] = b;
T[0, 14, 10] = a;
T[0, 14, 15] = b;

T[0, 15, 16] = 1;
T[0, 16, 16] = 1;

# down (a = 1)

T[1, 0, 0] = b;
T[1, 0, 4] = a;
T[1, 0, 1] = b;

T[1, 1, 0] = b;
T[1, 1, 5] = a;
T[1, 1, 2] = b;

T[1, 2, 1] = b;
T[1, 2, 6] = a;
T[1, 2, 3] = b;

T[1, 3, 2] = b;
T[1, 3, 7] = a;
T[1, 3, 3] = b;

T[1, 4, 4] = b;
T[1, 4, 8] = a;
T[1, 4, 5] = b;

T[1, 5, 4] = b;
T[1, 5, 9] = a;
T[1, 5, 6] = b;

T[1, 6, 5] = b;
T[1, 6, 10] = a;
T[1, 6, 7] = b;

T[1, 7, 6] = b;
T[1, 7, 11] = a;
T[1, 7, 7] = b;

T[1, 8, 8] = b;
T[1, 8, 12] = a;
T[1, 8, 9] = b;

T[1, 9, 8] = b;
T[1, 9, 13] = a;
T[1, 9, 10] = b;

T[1, 10, 9] = b;
T[1, 10, 14] = a;
T[1, 10, 11] = b;

T[1, 11, 10] = b;
T[1, 11, 15] = a;
T[1, 11, 11] = b;

T[1, 12, 12] = a + b;
T[1, 12, 13] = b;

T[1, 13, 12] = b;
T[1, 13, 13] = a;
T[1, 13, 14] = b;

T[1, 14, 13] = b;
T[1, 14, 14] = a;
T[1, 14, 15] = b;

T[1, 15, 16] = 1;
T[1, 16, 16] = 1;

# left (a = 2)

T[2, 0, 0] = a + b;
T[2, 0, 4] = b;

T[2, 1, 1] = b;
T[2, 1, 0] = a;
T[2, 1, 5] = b;

T[2, 2, 2] = b;
T[2, 2, 1] = a;
T[2, 2, 6] = b;

T[2, 3, 3] = b;
T[2, 3, 2] = a;
T[2, 3, 7] = b;

T[2, 4, 0] = b;
T[2, 4, 4] = a;
T[2, 4, 8] = b;

T[2, 5, 1] = b;
T[2, 5, 4] = a;
T[2, 5, 9] = b;

T[2, 6, 2] = b;
T[2, 6, 5] = a;
T[2, 6, 10] = b;

T[2, 7, 3] = b;
T[2, 7, 6] = a;
T[2, 7, 11] = b;

T[2, 8, 4] = b;
T[2, 8, 8] = a;
T[2, 8, 12] = b;

T[2, 9, 5] = b;
T[2, 9, 8] = a;
T[2, 9, 13] = b;

T[2, 10, 6] = b;
T[2, 10, 9] = a;
T[2, 10, 14] = b;

T[2, 11, 7] = b;
T[2, 11, 10] = a;
T[2, 11, 15] = b;

T[2, 12, 8] = b;
T[2, 12, 12] = a + b;

T[2, 13, 9] = b;
T[2, 13, 12] = a;
T[2, 13, 13] = b;

T[2, 14, 10] = b;
T[2, 14, 13] = a;
T[2, 14, 14] = b;

T[2, 15, 16] = 1;
T[2, 16, 16] = 1;

# right (a = 3)

T[3, 0, 0] = b;
T[3, 0, 1] = a;
T[3, 0, 4] = b;

T[3, 1, 1] = b;
T[3, 1, 2] = a;
T[3, 1, 5] = b;

T[3, 2, 2] = b;
T[3, 2, 3] = a;
T[3, 2, 6] = b;

T[3, 3, 3] = a + b;
T[3, 3, 7] = b;

T[3, 4, 0] = b;
T[3, 4, 5] = a;
T[3, 4, 8] = b;

T[3, 5, 1] = b;
T[3, 5, 6] = a;
T[3, 5, 9] = b;

T[3, 6, 2] = b;
T[3, 6, 7] = a;
T[3, 6, 10] = b;

T[3, 7, 3] = b;
T[3, 7, 7] = a;
T[3, 7, 11] = b;

T[3, 8, 4] = b;
T[3, 8, 9] = a;
T[3, 8, 12] = b;

T[3, 9, 5] = b;
T[3, 9, 10] = a;
T[3, 9, 13] = b;

T[3, 10, 6] = b;
T[3, 10, 11] = a;
T[3, 10, 14] = b;

T[3, 11, 7] = b;
T[3, 11, 11] = a;
T[3, 11, 15] = b;

T[3, 12, 8] = b;
T[3, 12, 13] = a;
T[3, 12, 12] = b;

T[3, 13, 9] = b;
T[3, 13, 14] = a;
T[3, 13, 13] = b;

T[3, 14, 10] = b;
T[3, 14, 15] = a;
T[3, 14, 14] = b;

T[3, 15, 16] = 1;
T[3, 16, 16] = 1;

# Reward function: |A| x |S| array
R = -1 * np.ones([4, 17]);

# set rewards
R[:, 15] = 100;  # goal state
R[:, 9] = -70;  # bad state
R[:, 5] = -70;  # bad state
R[:, 16] = 0;  # end state

# #print(R)

# # # current_state = 15
# # up = 0
# # down = 1
# # left = 2
# # right = 3
# # discount = 0.95
















# # # V = R[current_action]
# # V = np.zeros(17)

# # #compute value for each action
# # # action up a = 0
# # # R(s,a,s') + gamma*Value
# # # print (T[up, current_state] * (R[up] + discount * V))
# # # print(T[up, current_state])
# # # print(R[up][current_state] + discount * R[up])

# # count = 0
# # policy = np.zeros(17)
# # while count < np.inf:
# #     tempV = np.zeros(17)
# #     for current_state in range(17):
# #         up_value = sum(T[up, current_state] * (R[up][current_state] + discount * V))
# #         down_value = sum(T[down, current_state] * (R[down][current_state] + discount * V))
# #         left_value = sum(T[left, current_state] * (R[left][current_state] + discount * V))
# #         right_value = sum(T[right, current_state] * (R[right][current_state] + discount * V))

# #         action_list = [up_value, down_value, left_value, right_value]
# #         tempV[current_state] = max(action_list)
# #         policy[current_state] = action_list.index(max(action_list))
# #     # print(tempV)
# #     # print(V)
# #     count += 1
# #     if False not in np.isclose(tempV, V, rtol=0.01):
# #         break
# #     V = tempV

# # print(V[:4])
# # print(V[4:8])
# # print(V[8:12])
# # print(V[12:17])

# # print(policy[:4])
# # print(policy[4:8])
# # print(policy[8:12])
# # print(policy[12:17])
# # print(policy)

# # print (count)

# print("-----------------------------------------------------------------------------------------------")

# nStates = 17
# policy = np.zeros(nStates)
# V = np.zeros(nStates)
# iterID = 0
# discount = 0.95
# tolerance = 0.01
# nActions  = 4
# # print(R)

# while True:
#     iterID += 1
#     while True:
#         delta = 0
#         for current_state in range(nStates):
#             v = V[current_state]
#             action = int(policy[current_state])
#             V[current_state] = sum(T[action, current_state] * (R[action][current_state] + (discount * V)))
#             delta = max(delta, abs(v - V[current_state]))
#         if delta < tolerance:
#             break

#     print(iterID)
#     print(V[:4])
#     print(V[4:8])
#     print(V[8:12])
#     print(V[12:17])

#     policy_stable = True
#     for current_state in range(nStates):
#         old_action = int(policy[current_state])
#         # pick action that maximizes value
#         max_value = -np.inf
#         selected_action = old_action
#         for action in range(nActions):
#             value = sum(T[action, current_state] * (R[action][current_state] + (discount * V)))
#             if value > max_value:
#                 max_value = value
#                 selected_action = action
#         policy[current_state] = selected_action
#         if old_action != policy[current_state]:
#             policy_stable = False
#     if policy_stable is True:
#         break

# print(iterID)
# print(policy)

print(T)
print(R)