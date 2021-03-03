import numpy as np
import pandas as pd
# from sim_funs import find_state, define_primitive_actions
# import copy


class Option():
    def __init__(self, params):
        self.s_init = params["s_init"]
        self.s_term = params["s_term"]
        self.pi = params["pi"]


class PrimitiveAction(Option):
    pass


def softmax(Q, tau):
    """Compute softmax values for each sets of scores in x."""
    e_Q = np.exp(Q/tau)
    return e_Q / e_Q.sum(axis=0)


class Agent():
    def __init__(self, params, env):
        self.awake = False

        self.s_hist = []  # the agent's state at t-2

        # self.has_history = has_history
        # env.state_history = ["BO"]
        self.s_origin = None

        self.options = params["options"]

        # self.action_history = []

        self.r = None

        self.alpha = params["alpha"]  # learning rate
        self.gamma = params["gamma"]  # discounting factor

        self.selectionPolicy = params["policy"]
        if self.selectionPolicy == "e-greedy":
            self.eps = params["epsilon"]
        elif self.selectionPolicy == "softmax":
            self.tau = params["tau"]

        self.Q, optionNames = None, []
        for o_lbl, o in self.options.items():
            if self.Q is None:
                self.Q = np.array(o.s_init, dtype=float)
            else:
                self.Q = np.vstack((self.Q, o.s_init))
            optionNames.append(o_lbl)

        self.Q = pd.DataFrame(
            self.Q, columns=env.states.keys(), index=optionNames
        )

        self.activeOptions = []

    # def endowHistory(self, num_states):
    #     full_Q = []
    #     for i in range(num_states):
    #         full_Q.append(self.Q[:])
    #
    #     self.Q = np.array(full_Q)

    def wakeUp(self, env):
        self.awake = True

        self.s_origin = env.state["coords"]

        self.selectOption(env)

    def selectOption(self, env):
        '''
        Selects an option from the policy given by the lowest level currently
        active option. If no options are currently active, one will be selected
        from the agent's Q matrix.

        Expects     env - an object of class Environment

        '''
        print("selecting")

        if self.activeOptions == []:
            pi = self.Q
        else:
            pi = self.options[self.activeOptions[-1]["label"]].pi

        Q_sa = pi.loc[:, env.state["label"]].squeeze()

        if self.selectionPolicy == "greedy":
            Q_max = Q_sa.transform(lambda x: x == x.max()).astype('bool')
            maxRow = Q_sa.loc[Q_max]
            if len(maxRow) > 1:
                maxRow = maxRow.sample()
            choice = maxRow.index[0]

        elif self.selectionPolicy == "e-greedy":
            if np.random.random() < self.eps:
                choice = np.random.choice(np.where(Q_sa > 0)[0])

        elif self.selectionPolicy == "softmax":
            Q_sa = softmax(Q_sa, self.tau)
            choice = Q_sa.sample(weights=Q_sa).index[0]

        self.activeOptions.append(
            {"label": choice, "stateInitialised": env.state}
        )

        if isinstance(self.options[choice], PrimitiveAction) is False:
            self.selectOption(env)

        else:
            self.move(env)

    # def update_action_history(self, choice):
    #     self.action_history.append(self.option_lbls[choice])

    def move(self, env):
        '''
        Moves the agent according to the lowest level (primitive) currently
        active option.

        Expects     env - an object of class Environment

        '''
        action = self.activeOptions[-1]["label"]
        print("moving " + action)

        # Compute and perform the shifts in x and y given by the chosen
        # primitive action:
        x_shift = 1
        if action[1] == "W":
            x_shift *= -1

        y_shift = 1
        if action[0] == "S":
            y_shift *= -1

        env.state["coords"][0] += x_shift
        env.state["coords"][1] += y_shift

        # If the agent has returned to the origin, ensure that the
        # z-coordinates correspond to the z given by the arrangement of the
        # current trial:
        if env.state["coords"][:2] == [0, 0]:
            env.state["coords"] = self.s_origin[:]
        else:
            env.state["coords"][2] = 0

        env.update()

        # Find and store the index of the new state:
        # self.s_i = find_state(env.state, env)
        # self.update_state_history(env)

        # If the new state is a sub-goal, record visitation:
        if env.state["label"] == env.SG:
            self.SG_visited = True

        # self.step_counter += 1

        self.collectReward(env)

    # def update_state_history(self, env):
    #     env.state_history.append(find_state(env.state, env, value="label"))

    def collectReward(self, env):
        '''
        Collects reward from the environment if available. If the task is
        hierarchical, reward at the goal location is contingent upon the agent
        having first visited the sub-goal, and so this must be evaluated prior
        to award.

        Expects        env - an object of class Environment
                 task_mode - a str describing the mode of the current task

        '''
        print("collecting reward")
        self.r += env.deliverReward(self.SG_visited)

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
        print("checking for termination")

        controlOption = self.options[self.activeOptions[-1]["label"]]

        if controlOption.s_term[env.state["label"]][0] == 1:
            self.terminateOption(env)
        else:
            if isinstance(controlOption, PrimitiveAction) is False:
                self.selectOption(env)
            else:
                self.move(env)

    def terminateOption(self, env):
        '''
        Terminates the lowest level currently active option, and stores its
        name for later evaluation.

        '''
        print("terminating")
        self.terminatedOption = self.activeOptions.pop(-1)

        # If active_policies is empty, we update Q:
        if self.activeOptions == []:
            self.update_Q(env)
        else:
            self.checkForTermination(env)

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
        print("updating Q")
        # s_hist = find_state(self.s_hist, env)
        s_prev = self.terminatedOption["stateInitialised"]["label"]
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

        if self.r == 0:
            self.selectOption(env)
        else:
            self.sleep()

    def sleep(self):
        '''
        Reset the agent prior to the start of a new episode.

        Expects        env - an object of class Environment
                   SG_side - a str describing the SG-location of the next trial

        '''
        self.awake = False
        self.SG_visited = False
        self.r = 0
