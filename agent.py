import numpy as np
import pandas as pd
from sim_funs import find_state, define_primitive_actions
import copy


class Option():
    def __init__(self, params):
        self.s_init = params["s_init"]
        self.s_term = params["s_term"]
        self.pi = params["pi"]


class PrimitiveAction(Option):
    pass


class Agent():
    def __init__(self, params, env):
        # self.s = None  # state of the agent in cartesian coords
        # self.s_i = 0  # the index of the agent's current state
        # self.s_prev = self.state  # the agent's state at t-1
        self.s_hist = []  # the agent's state at t-2

        # self.has_history = has_history
        # self.state_history = ["BO"]
        self.s_origin = None

        self.options = params["options"]

        # self.action_lbls = action_lbls  # labels of all primitive actions
        # self.num_actions = len(self.options)  # number of primitive actions
        # self.option_lbls = copy.deepcopy(action_lbls)  # labels of all available options
        # self.action_history = []

        self.r = None  # the reward experienced by the agent in it's current state

        self.alpha = params["alpha"]  # learning rate
        self.gamma = params["gamma"]  # discounting factor

        self.selectionPolicy = params["policy"]
        if self.selection_policy == "e-greedy":
            self.eps = params["epsilon"]
        elif self.selection_policy == "softmax":
            self.tau = params["tau"]
        # self.step_counter = 0

        self.Q, optionNames = None, []
        for o in self.options:
            if self.Q is None:
                self.Q = np.array(o.s_init, dtype=float)
            else:
                self.Q = np.vstack((self.Q, o.s_init))
            optionNames.append(o.label)

        self.Q = pd.DataFrame(
            self.Q, columns=env.states.keys, index=optionNames
        )

        # self.beta = []  # holds the termination conditions for each option
        # self.o_policies = []  # holds the policy for each option
        self.activeOptions = []  # holds all currently active policies for
        # all selected options in hierarchical order (i.e., the last item in
        # the list will be the policy tied to the lowest level currently active
        # option)
        # self.under_primitive_control = False  # indicates whether the lowest
        # level active option has direct behavioural control (i.e., is one of
        # the available primitive actions)
        # self.under_Q_control = True  # indicates whether no options are active,
        # requiring deferral to the Q-matrix to proceed

        # self.s_init = None  # hold the value of the state in which the highest
        # level active option was initiated (for later Q-updating)
        # self.terminated_option = None  # holds the name of the most recently
        # terminated option (for Q-updating)

        # self.termination_reached = False  # indicates whether the agent has
        # reached a termination state (ie a state in which the episode ends)

    # def init_primitive_actions(self, task_mode):
    #     '''
    #     Initializes the agent's (four) primitive actions (ie NE, SE, SW, NW).
    #     These are stored in Q, where each row defines the initiation states of
    #     each available action, o_policies, where the first set of option
    #     policies are implement the available primitive actions, and beta, which
    #     defines the termination conditions of each of the available actions.
    #
    #     Expects     task_mode - a str indicating the mode of the task (flat or
    #                             hierarchical)
    #
    #     '''
    #     s_init, s_term, pi = define_primitive_actions(
    #         self.action_lbls, task_mode
    #     )
    #
    #     for a in self.action_lbls:
    #         if self.Q is None:
    #             self.Q = np.array(s_init[a], dtype=float)
    #         else:
    #             self.Q = np.vstack((self.Q, s_init[a]))
    #
    #         self.beta.append(s_term[a])
    #
    #         self.o_policies.append(pi[a])

    # def endowHistory(self, num_states):
    #     full_Q = []
    #     for i in range(num_states):
    #         full_Q.append(self.Q[:])
    #
    #     self.Q = np.array(full_Q)

    # def add_option(self, option):
    #     '''
    #     Adds an option to the agent's repetoir of actions. This includes adding
    #     a row to Q (which defines the option's initiation states), a row to
    #     beta (which defines the option's termination states), a label to
    #     option_lbls, and a policy to o_policies.
    #
    #     Expects     option - an object of class Option, which includes s_term,
    #                          s_init, pi, and label as properties
    #
    #     '''
    #
    #     self.Q = np.vstack((self.Q, option.s_init))
    #     self.beta.append(option.s_term)
    #     self.o_policies.append(option.pi)
    #     self.option_lbls.append(option.label)

    def wakeUp(self, env):
        self.s_origin = env.state["coords"]

        self.selectOption(env)

    def selectOption(self, env):
        '''
        Selects an option from the policy given by the lowest level currently
        active option. If no options are currently active, one will be selected
        from the agent's Q matrix.

        Expects     env - an object of class Environment

        '''
        # e_threshold = self.e_end + (self.e_start - self.e_end) * \
        #     np.exp(-1. * self.step_counter / self.e_decay)

        if self.activeOptions == []:
            pi = self.Q
        else:
            pi = self.options[self.activeOptions[-1]["label"]].pi

        Q_sa = pi.loc[:, env.state]  # this defines the Q-values held by the agent
        # for all state-action pairs from it's current state

        # if np.random.random() < e_threshold:  # if under e-greedy, evaluate e
        # choice = np.random.choice(np.where(Q_s_a > 0)[0])
        # else:  # otherwise select the greedy option
        Q_max = Q_sa.transform(lambda x: x == x.max()).astype('bool')
        maxRow = Q_s_a.loc[Q_max]
        if len(maxRow) > 1:
            maxRow = maxRow.sample()
        choice = maxRow.index[0]

        self.activeOptions.append(
            {"label": choice, "stateInitialised": env.state}
        )

        # If the agent is currently under Q-control, it must store it's current
        # state in memory such that it can update the value of the chosen
        # option upon its termination. As such, we store the index of the
        # current state before setting under_Q_control to False.
        # if self.under_Q_control:
        #     self.s_init = self.s_i
        #     self.under_Q_control = False
        #     self.update_action_history(choice)

        # If the chosen policy corresponds to one of the available primitive
        # actions, the agent needs to initiate movement, which requires setting
        # under_primitive_control to True.
        if isinstance(self.options[choice], PrimitiveAction) is False:
            # self.move()
            # else:
            self.selectOption()

        else:
            self.move()

    # def update_action_history(self, choice):
    #     self.action_history.append(self.option_lbls[choice])

    def move(self, env):
        '''
        Moves the agent according to the lowest level (primitive) currently
        active option.

        Expects     env - an object of class Environment

        '''
        action = self.activeOptions[-1]  # define the index of the
        # primitive action given by the currently active primitive option

        # Compute and perform the shifts in x and y given by the chosen
        # primitive action:
        # if (not np.array_equal(self.s_prev, self.state)) and (self.has_history):
        #     self.s_hist = self.s_prev[:]
        # self.s_prev = self.state[:]

        x_shift = 1
        if action[1] == "W":
            x_shift *= -1

        y_shift = 1
        if action[0] == "S":
            y_shift *= -1

        env.state["coords"][0] += x_shift
        env.state["coords"][1] += y_shift
        env.update()

        # If the agent has returned to the origin, ensure that the
        # z-coordinates correspond to the z given by the arrangement of the
        # current trial:
        if self.state[:2] == [0, 0]:
            self.state = self.s_origin[:]
        else:
            self.state[2] = 0

        # Find and store the index of the new state:
        # self.s_i = find_state(self.state, env)
        # self.update_state_history(env)

        # If the new state is a sub-goal, record visitation:
        if env.findState(env.state) == env.SG:
            self.SG_visited = True

        self.step_counter += 1

        self.collectReward(env)

    # def update_state_history(self, env):
    #     self.state_history.append(find_state(self.state, env, value="label"))

    def collectReward(self, env):
        '''
        Collects reward from the environment if available. If the task is
        hierarchical, reward at the goal location is contingent upon the agent
        having first visited the sub-goal, and so this must be evaluated prior
        to award.

        Expects        env - an object of class Environment
                 task_mode - a str describing the mode of the current task

        '''
        self.r = env.deliverReward(self.SG_visited)

        # if self.r == 1:
        #     self.termination_reached = True

        # self.update_Q(env)
        self.checkForTermination(env)

    def checkForTermination(self, env):
        '''
        Checks whether any currently active options should be terminated given
        movement into a new state and the list of all currently active
        policies. This occurs by iterating through active_policies in reverse
        order, as the lowest-level active policies occupy the end of the list,
        and will be the first to terminate. If a termination condition is
        satisfied, the method triggers termination by calling
        terminate_control_policy.

        '''
        if self.activeOptions[-1]["label"].s_term[env.state["label"]][0] == 1:
            self.terminateOption()

    def terminateOption(self, env):
        '''
        Terminates the lowest level currently active option, and stores its
        name for later evaluation.

        '''
        self.terminatedOption = self.active_policies.pop(-1)

        # If active_policies is empty, control of the agent must pass to Q:
        if self.activeOptions == []:
            self.update_Q(env)
        #
        # # If the lowest level currently active policy is non-primitive, a
        # # primitive policy must be sought out:
        # elif self.active_policies[-1] >= self.num_actions:
        #     self.under_primitive_control = False

    def update_Q(self, env):
        '''
        Upon termination of all active policies, evaluate the Q-value
        associated with the highest level and most recently active policy (ie
        the policy that was terminated prior to control being handed to the
        Q-matrix) and the state in which it was initiated.

        Expects        env - an object of class Environment

        '''
        # s_hist = find_state(self.s_hist, env)
        s_prev = self.terminatedOption["stateInitialised"]
        s_curr = env.state["label"]
        o = self.terminatedOption["label"]

        # if self.has_history:
        #     delta = self.alpha*(self.r +
        #                         self.gamma * np.max(
        #                             self.Q[s_prev, :, s]) -
        #                         self.Q[s_hist, o, s_prev])
        #
        #     self.Q[s_hist, o, s_prev] += delta

        # else:
        # if len(self.option_lbls) > 4:
        #     s_prev = self.s_init

        delta = self.alpha * (
            self.r +
            self.gamma * np.max(self.Q.loc[:, s_curr]) - self.Q.loc[o, s_prev]
        )

        self.Q.loc[o, s_prev] += delta

        self.selectOption()

    def reset(self, SG_side, env, task_mode):
        '''
        Reset the agent prior to the start of a new episode.

        Expects        env - an object of class Environment
                   SG_side - a str describing the SG-location of the next trial

        '''
        self.r = 0

    #     self.action_history = []
    #     self.state_history = ["BO"]
    #     self.s_prev = 0
    #
    #     self.under_Q_control = True
    #     self.under_primitive_control = False
    #
    #     if task_mode is "hierarchical":
    #         if SG_side is "L":
    #             self.state = [0, 0, 0]
    #             self.origin = [0, 0, 0]
    #         else:
    #             self.state = [0, 0, 1]
    #             self.origin = [0, 0, 1]
    #     else:
    #         self.state = self.origin = [0, 0, 0]
    #
    #     self.s_i = find_state(self.state, env)
    #     self.s_prev = self.state[:]
    #     self.s_hist = self.state[:]
    #
        self.SG_visited = False
