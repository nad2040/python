import numpy as np

ACTION_SPACE = ('U', 'D', 'L', 'R')
REWARD_SPACE = (-1,0)

class Grid:  # Environment
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, terminal_rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.terminal_rewards = terminal_rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def reset(self):
        # put agent back in start position
        self.i = 2
        self.j = 0
        return (self.i, self.j)

    def get_next_state(self, s, a):
        # this answers: where would I end up if I perform action 'a' in state 's'?
        i, j = s[0], s[1]

        # if this action moves you somewhere else, then it will be in this dictionary
        if a in self.actions[(i, j)]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'R':
                j += 1
            elif a == 'L':
                j -= 1
        return i, j

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return a reward (if any)
        return self.terminal_rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert (self.current_state() in self.all_states())

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.terminal_rewards.keys())


def standard_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    g = Grid(4, 4, (2, 1))
    terminal_rewards = {(0, 0): -1, (3, 3): -1}
    actions = {
            (0, 1): ('L', 'D', 'R'),
            (0, 2): ('L', 'D', 'R'),
            (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'),
            (1, 1): ('U', 'L', 'D', 'R'),
            (1, 2): ('U', 'L', 'D', 'R'),
            (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'),
            (2, 1): ('U', 'L', 'D', 'R'),
            (2, 2): ('U', 'L', 'D', 'R'),
            (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'),
            (3, 1): ('U', 'L', 'R'),
            (3, 2): ('U', 'L', 'R')
            }
    g.set(terminal_rewards, actions)

    return g


grid = standard_grid()
print(grid.all_states())
print(grid.actions)
print(grid.terminal_rewards)

def transition_probability_and_reward(grid):
    transition = {}
    for s in grid.actions.keys():
        for a in ACTION_SPACE:
            distribution = {}
            for s2 in grid.all_states():
                for r in REWARD_SPACE:
                    # this code is special for the gridworld case
                    # only one possible next state and one possible reward per step
                    if s2 == grid.get_next_state(s, a) and r == -1:
                        distribution[(s2, grid.terminal_rewards.get(s2, r))] = 1
            transition[(s,a)] = distribution
    return transition


def print_values(V, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print()
    print()


def print_policy(P, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print()
    print()


THRESHOLD = 1e-3
GAMMA = 1

P = transition_probability_and_reward(grid)


def initial_policy(grid):
    policy = {}
    for s in grid.actions.keys():
        policy[s] = { a:1/len(ACTION_SPACE) for a in ACTION_SPACE }
    return policy


def policy_evaluation(grid, policy, value_iteration=False):
    V = {}
    for s in grid.all_states():
        V[s] = 0
    iterations = 0
    while True:
        delta = 0
        for s in grid.actions.keys():
            v = V[s]

            updated_value = 0
            for a in ACTION_SPACE:
                action_value = 0
                for s2 in grid.all_states():
                    for r in REWARD_SPACE:
                        action_value += P[(s,a)].get((s2, r), 0) * (r + GAMMA * V[s2])
                updated_value += policy[s][a] * action_value

            V[s] = updated_value

            delta = max(delta, abs(v - V[s]))

        print_values(V, grid)
        iterations+=1
        if delta < THRESHOLD or value_iteration:
            break
    print(f"iterations: {iterations}")
    return V


def policy_improvement(grid, policy, state_valuation):
    policy_stable = True

    for s in grid.actions.keys():
        old_action = policy[s]

        action_values = []
        for a in ACTION_SPACE:
            s2 = grid.get_next_state(s, a)
            # in this deterministic policy, state value is action value?
            action_values.append((grid.terminal_rewards.get(s2,state_valuation[s2]),a))
        policy[s] = max(action_values)[1]

        if old_action != policy[s]:
            policy_stable = False

    return policy, policy_stable


def policy_iteration(grid):
    policy = initial_policy(grid)
    state_values = {}

    print_policy(policy, grid)

    while True:
        print("policy evaluation")
        state_values = policy_evaluation(grid, policy)
        print_values(state_values, grid)

        print("policy improvement")
        policy, policy_stable = policy_improvement(grid, policy, state_values)
        print_policy(policy, grid)

        if policy_stable:
            break

    return state_values, policy


def value_iteration(grid):
    state_values = {}
    for s in grid.all_states():
        state_values[s] = 0

    iterations = 0
    while True:
        delta = 0
        for s in grid.actions.keys():
            v = state_values[s]

            action_values = []
            for a in ACTION_SPACE:
                s2 = grid.get_next_state(s, a)
                action_value = 0
                for s2 in grid.all_states():
                    for r in REWARD_SPACE:
                        action_value += P[(s,a)].get((s2, r), 0) * (r + GAMMA * state_values[s2])
                action_values.append(action_value)
            state_values[s] = max(action_values)

            delta = max(delta, abs(v - state_values[s]))

        print_values(state_values, grid)
        iterations+=1
        if delta < THRESHOLD:
            break
    print(f"iterations: {iterations}")

    policy = {}
    for s in grid.actions.keys():
        action_values = []
        for a in ACTION_SPACE:
            s2 = grid.get_next_state(s, a)
            action_value = 0
            for s2 in grid.all_states():
                for r in REWARD_SPACE:
                    action_value += P[(s,a)].get((s2, r), 0) * (r + GAMMA * state_values[s2])
            action_values.append(action_value)

        policy[s] = ACTION_SPACE[action_values.index(max(action_values))]

    print_policy(policy, grid)


# policy_iteration(grid)
#
# policy = initial_policy(grid)
# print(policy)
# state_values = policy_evaluation(grid, policy)
#
value_iteration(grid)
