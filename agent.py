import numpy as np
from sim_funs import find_state, define_primitive_actions
import copy


class Agent():
    def __init__(self, alpha, gamma, action_lbls, policy, epsilon, has_history):
        self.state = None  # state of the agent in cartesian coords
        self.s_i = 0  # the index of the agent's current state
        self.s_prev = self.state  # the agent's state at t-1
        self.s_hist = self.state  # the agent's state at t-2
        self.has_history = has_history
        self.state_history = ["BO"]
        self.origin = None

        self.action_lbls = action_lbls  # labels of all primitive actions
        self.num_actions = len(self.action_lbls)  # number of primitive actions
        self.option_lbls = copy.deepcopy(action_lbls)  # labels of all available options
        self.action_history = []

        self.r = 0  # the reward experienced by the agent in it's current state

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discounting factor

        self.selection_policy = policy
        if self.selection_policy == "e-greedy":
            self.e_start = epsilon["start"]
            self.e_end = epsilon["end"]
            self.e_decay = epsilon["decay"]
        elif self.selection_policy == "greedy":
            self.e_start = 0
            self.e_end = 0
            self.e_decay = 0
        self.step_counter = 0

        self.Q = None  # the top-level Q-matrix, which holds Q-values for each
        # state-option pair (also holds the initiation conditions for each)
        # option

        self.beta = []  # holds the termination conditions for each option
        self.o_policies = []  # holds the policy for each option
        self.active_policies = []  # holds all currently active policies for
        # all selected options in hierarchical order (i.e., the last item in
        # the list will be the policy tied to the lowest level currently active
        # option)
        self.under_primitive_control = False  # indicates whether the lowest
        # level active option has direct behavioural control (i.e., is one of
        # the available primitive actions)
        self.under_Q_control = True  # indicates whether no options are active,
        # requiring deferral to the Q-matrix to proceed

        self.s_init = None  # hold the value of the state in which the highest
        # level active option was initiated (for later Q-updating)
        self.terminated_option = None  # holds the name of the most recently
        # terminated option (for Q-updating)

        self.termination_reached = False  # indicates whether the agent has
        # reached a termination state (ie a state in which the episode ends)

    def init_primitive_actions(self, task_mode):
        '''
        Initializes the agent's (four) primitive actions (ie NE, SE, SW, NW).
        These are stored in Q, where each row defines the initiation states of
        each available action, o_policies, where the first set of option
        policies are implement the available primitive actions, and beta, which
        defines the termination conditions of each of the available actions.

        Expects     task_mode - a str indicating the mode of the task (flat or
                                hierarchical)

        '''
        s_init, s_term, pi = define_primitive_actions(
            self.action_lbls, task_mode
        )

        for a in self.action_lbls:
            if self.Q is None:
                self.Q = np.array(s_init[a], dtype=float)
            else:
                self.Q = np.vstack((self.Q, s_init[a]))

            self.beta.append(s_term[a])

            self.o_policies.append(pi[a])

    def endow_history(self, num_states):
        full_Q = []
        for i in range(num_states):
            full_Q.append(self.Q[:])

        self.Q = np.array(full_Q)

    def add_option(self, option):
        '''
        Adds an option to the agent's repetoir of actions. This includes adding
        a row to Q (which defines the option's initiation states), a row to
        beta (which defines the option's termination states), a label to
        option_lbls, and a policy to o_policies.

        Expects     option - an object of class Option, which includes s_term,
                             s_init, pi, and label as properties

        '''

        self.Q = np.vstack((self.Q, option.s_init))
        self.beta.append(option.s_term)
        self.o_policies.append(option.pi)
        self.option_lbls.append(option.label)

    def select_option(self, env):
        '''
        Selects an option from the policy given by the lowest level currently
        active option. If no options are currently active, one will be selected
        from the agent's Q matrix.

        Expects     env - an object of class Environment

        '''
        e_threshold = self.e_end + (self.e_start - self.e_end) * \
            np.exp(-1. * self.step_counter / self.e_decay)

        if self.under_Q_control:
            if self.has_history:
                pi = self.Q[find_state(self.s_prev, env), :, :]
            else:
                pi = self.Q
        else:
            pi = self.o_policies[self.active_policies[-1]]

        Q_s_a = pi[:, self.s_i]  # this defines the Q-values held by the agent
        # for all state-action pairs from it's current state

        if np.random.random() < e_threshold:  # if under e-greedy, evaluate e
            choice = np.random.choice(np.where(Q_s_a > 0)[0])
        else:  # otherwise select the greedy option
            choice = np.where(Q_s_a == max(Q_s_a))[0]
            if len(choice) > 1:
                choice = np.random.choice(choice)
            else:
                choice = choice[0]

        # If the agent is currently under Q-control, it must store it's current
        # state in memory such that it can update the value of the chosen
        # option upon its termination. As such, we store the index of the
        # current state before setting under_Q_control to False.
        if self.under_Q_control:
            self.s_init = self.s_i
            self.under_Q_control = False
            self.update_action_history(choice)

        # If the chosen policy corresponds to one of the available primitive
        # actions, the agent needs to initiate movement, which requires setting
        # under_primitive_control to True.
        if choice < self.num_actions:
            self.under_primitive_control = True

        self.active_policies.append(choice)

    def update_action_history(self, choice):
        self.action_history.append(self.option_lbls[choice])

    def move(self, env):
        '''
        Moves the agent according to the lowest level (primitive) currently
        active option.

        Expects     env - an object of class Environment

        '''
        self.a_i = self.active_policies[-1]  # define the index of the
        # primitive action given by the currently active primitive option

        # Compute and perform the shifts in x and y given by the chosen
        # primitive action:
        if (not np.array_equal(self.s_prev, self.state)) and (self.has_history):
            self.s_hist = self.s_prev[:]
        self.s_prev = self.state[:]

        x_shift = 1
        if self.action_lbls[self.a_i][1] == "W":
            x_shift *= -1

        y_shift = 1
        if self.action_lbls[self.a_i][0] == "S":
            y_shift *= -1

        self.state[0] += x_shift
        self.state[1] += y_shift

        # If the agent has returned to the origin, ensure that the
        # z-coordinates correspond to the z given by the arrangement of the
        # current trial:
        if self.state[:2] == [0, 0]:
            self.state = self.origin[:]
        else:
            self.state[2] = 0

        # Find and store the index of the new state:
        self.s_i = find_state(self.state, env)
        self.update_state_history(env)

        # If the new state is a sub-goal, record visitation:
        if find_state(self.state, env, value="label") == env.SG:
            self.SG_visited = True

        self.step_counter += 1

    def update_state_history(self, env):
        self.state_history.append(find_state(self.state, env, value="label"))

    def check_for_termination(self):
        '''
        Checks whether any currently active options should be terminated given
        movement into a new state and the list of all currently active
        policies. This occurs by iterating through active_policies in reverse
        order, as the lowest-level active policies occupy the end of the list,
        and will be the first to terminate. If a termination condition is
        satisfied, the method triggers termination by calling
        terminate_control_policy.

        '''

        for o in reversed(self.active_policies):
            if self.beta[o][self.s_i] == 1:
                self.terminate_control_policy()

    def terminate_control_policy(self):
        '''
        Terminates the lowest level currently active option, and stores its
        name for later evaluation.

        '''

        self.terminated_option = self.active_policies.pop(-1)

        # If active_policies is empty, control of the agent must pass to Q:
        if not self.active_policies:
            self.under_Q_control = True
            self.under_primitive_control = False

        # If the lowest level currently active policy is non-primitive, a
        # primitive policy must be sought out:
        elif self.active_policies[-1] >= self.num_actions:
            self.under_primitive_control = False

    def collect_reward(self, env, task_mode):
        '''
        Collects reward from the environment if available. If the task is
        hierarchical, reward at the goal location is contingent upon the agent
        having first visited the sub-goal, and so this must be evaluated prior
        to award.

        Expects        env - an object of class Environment
                 task_mode - a str describing the mode of the current task

        '''

        if task_mode == "hierarchical":
            if self.SG_visited:
                self.r = env.pR[self.s_i]
            else:
                self.r = 0
        elif task_mode == "flat":
            self.r = env.pR[self.s_i]

        if self.r == 1:
            self.termination_reached = True

    def update_Q(self, env):
        '''
        Upon termination of all active policies, evaluate the Q-value
        associated with the highest level and most recently active policy (ie
        the policy that was terminated prior to control being handed to the
        Q-matrix) and the state in which it was initiated.

        Expects        env - an object of class Environment

        '''
        s_hist = find_state(self.s_hist, env)
        s_prev = find_state(self.s_prev, env)
        s = find_state(self.state, env)
        o = self.terminated_option

        if self.has_history:
            delta = self.alpha*(self.r +
                                self.gamma * np.max(
                                    self.Q[s_prev, :, s]) -
                                self.Q[s_hist, o, s_prev])

            self.Q[s_hist, o, s_prev] += delta

        else:
            if len(self.option_lbls) > 4:
                s_prev = self.s_init

            delta = self.alpha*(self.r +
                                self.gamma * np.max(self.Q[:, s]) -
                                self.Q[o, s_prev])

            self.Q[o, s_prev] += delta

    def reset(self, SG_side, env, task_mode):
        '''
        Reset the agent prior to the start of a new episode.

        Expects        env - an object of class Environment
                   SG_side - a str describing the SG-location of the next trial

        '''
        self.r = 0

        self.action_history = []
        self.state_history = ["BO"]
        self.s_prev = 0

        self.under_Q_control = True
        self.under_primitive_control = False

        if task_mode is "hierarchical":
            if SG_side is "L":
                self.state = [0, 0, 0]
                self.origin = [0, 0, 0]
            else:
                self.state = [0, 0, 1]
                self.origin = [0, 0, 1]
        else:
            self.state = self.origin = [0, 0, 0]

        self.s_i = find_state(self.state, env)
        self.s_prev = self.state[:]
        self.s_hist = self.state[:]

        self.SG_visited = False


class Option():
    def __init__(self, option_params):
        self.label = option_params["label"]

        self.s_init = option_params["s_init"]
        self.s_term = option_params["s_term"]

        self.pi = option_params["pi"]
