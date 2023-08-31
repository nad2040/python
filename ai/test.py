states = [(i,j) for i in range(4) for j in range(4)]
V = {s: 0 for s in states}

def terminal(state):
  return state == (0, 0) or state == (3,3)

def up(state):
  return (max(0, state[0]-1), state[1])

def down(state):
  return (min(3, state[0]+1), state[1])

def left(state):
  return (state[0], max(0, state[1]-1))

def right(state):
  return (state[0], min(3, state[1]+1))

# policy evaluation, sweep with old and update to new
'''
Vnew = {}
for i in range(200):
  Vnew.clear()
  delta = 0
  for state in states:
    if (terminal(state)):
      Vnew[state] = V[state]
    else:
      Vnew[state] = 0.25 * ((-1 + V[up(state)]) + (-1 + V[down(state)]) + (-1 + V[left(state)]) + (-1 + V[right(state)]))
    delta = max(delta, abs(Vnew[state] - V[state]))
  V = Vnew.copy()
  print(i, delta, V)
  if delta < 1e-3:
    break
'''

import numpy as np

def toArray(v):
  a = np.zeros((4,4))
  for s in states:
    a[s[0]][s[1]] = v[s]
  return a


#policy iteration

#init
states = [(i,j) for i in range(4) for j in range(4)]
V = {s: 0 for s in states}
actions = [up, down, left, right]
import random
import time
random.seed(time.time())
pi = {s: random.choice(actions) for s in states}

while True:
  print("policy eval")
  i = 0
  while True:
    delta = 0
    for state in states:
      v = V[state]
      if not terminal(state):
        V[state] = (-1 + 0.9 * V[pi[state](state)])
      delta = max(delta, abs(V[state] - v))
    i += 1
    print(i, delta, '\n', toArray(V))
    if delta < 1e-3:
      break

  print("policy improvement")
  stable = True
  for state in states:
    if terminal(state):
      continue
    oldAction = pi[state]
    #pvs = [V[up(state)], V[down(state)], V[left(state)], V[down(state)]
    pvs = [V[action(state)] for action in actions] #omit adding constant reward
    pi[state] = actions[np.argmax(pvs)] #argmax(a)
    if oldAction != pi[state]:
      stable = False
  if stable:
    print("done with pi*:", pi)
    break

