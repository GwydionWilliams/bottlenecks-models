import numpy as np
import pandas as pd
# from sim_funs import find_state, define_primitive_actions
import copy


class Option():
    def __init__(self, params):
        self.s_init = params["s_init"]
        self.s_term = params["s_term"]
        self.pi = params["pi"]
        self.level = params["level"]


class PrimitiveAction(Option):
    pass


class ParameterError(Exception):
    pass


def softmax(Q, tau):
    """Compute softmax values for each sets of scores in x."""
    e_Q = np.exp(Q/tau)
    return (e_Q / e_Q.sum(axis=0)).fillna(1)


class Agent():
    def __init__(self, params, env):
        self.awake = False
        self.talkative = params["talkative"]

        self.s_hist = []  # the agent's state at t-2

        self.s_origin = None
        if "hierarchical" in params["class"]:
            if params["representsHistory"] is True:
                raise ParameterError(
                    "Hierarchical agents cannot represent history."
                )
            else:
                self.representsHistory = False
        else:
            self.representsHistory = params["representsHistory"]

        if "flat" in params["class"] and \
                params["selectionStrategy"] is "sequential":
            raise ParameterError(
                "Flat agents cannot employ sequential option selection."
            )
        else:
            self.selectionStrategy = params["selectionStrategy"]

        self.statesForAbstraction = params["statesForAbstraction"]

        self.options = params["options"]

        self.r = None

        self.alpha = params["alpha"]  # learning rate
        self.gamma = params["gamma"]  # discounting factor

        self.selectionPolicy = params["policy"]
        if self.selectionPolicy == "e-greedy":
            self.eps = params["epsilon"]
        elif "softmax" in self.selectionPolicy:
            self.tau = params["tau"]

            if "structured" in self.selectionPolicy:
                self.beta = params["beta"]
            else:
                self.beta = "NA"

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

        if self.representsHistory:
            origins = list(env.states.keys())[:2]
            Q_hist = {}
            for origin in origins:
                Q_hist.update({origin: copy.deepcopy(self.Q)})

            self.Q = Q_hist

        self.activeOptions = []

        self.stepCounter = 0

    def wakeUp(self, env):
        self.awake = True
        self.s_origin = copy.deepcopy(env.state)
        self.stepCounter = 0
        # self.selectOption(env)

        if self.talkative:
            print("starting trial at", self.s_origin["label"])

    def selectOption(self, env):
        '''
        Selects an option from the policy given by the lowest level currently
        active option. If no options are currently active, one will be selected
        from the agent's Q matrix.

        Expects     env - an object of class Environment

        '''
        # 1. Find Q_sa --------------------------------------------------------
        if self.activeOptions == []:
            if not self.representsHistory:
                pi = self.Q
            else:
                pi = self.Q[self.s_origin["label"]]

            isDeterministic = False

            if self.selectionStrategy is "sequential":
                level, optsAtLevel = self.findHighestLevelOptions(env)
                pi = pi.loc[optsAtLevel, ]

        else:
            pi = self.options[self.activeOptions[-1]["label"]].pi
            isDeterministic = True

        Q_sa = pi.loc[:, env.state["label"]]

        # 2. Select action from Q_sa ------------------------------------------
        # 2a. greedy selection
        if self.selectionPolicy == "greedy" or isDeterministic:
            Q_max = Q_sa.transform(lambda x: x == x.max()).astype('bool')
            maxRow = Q_sa.loc[Q_max]
            if len(maxRow) > 1:
                maxRow = maxRow.sample()
            choice = maxRow.index[0]

        # 2b. e-greedy selection
        elif self.selectionPolicy == "e-greedy":
            if np.random.random() < self.eps:
                choice = np.random.choice(np.where(Q_sa > 0)[0])

        # 2cd. structured-/softmax selection
        elif "softmax" in self.selectionPolicy:
            if "structured" in self.selectionPolicy:
                optLevels = np.unique([o.level for o in self.options.values()])
                taus = self.tau * \
                    self.beta ** (max(optLevels) - optLevels)
                tau = taus[level]

                if self.talkative:
                    print("selecting option from level", level,
                          "with temperature", tau)
            else:
                Q_sa = Q_sa.dropna()
                tau = self.tau

            Q_sa = softmax(Q_sa, tau)

            choice = Q_sa.sample(weights=Q_sa).index[0]

        self.activeOptions.append(
            {"label": choice, "stateInitialised": copy.deepcopy(env.state)}
        )

        # 3. Check if primitive
        if isinstance(self.options[choice], PrimitiveAction) is False:
            self.selectOption(env)

        elif self.talkative:
            print(
                "selected action hierarchy is: ",
                [o["label"] for o in self.activeOptions]
            )

    def findHighestLevelOptions(self, env):
        Q_s = self.Q.loc[:, env.state["label"]]
        Q_s = Q_s.dropna()
        optsAvailable = Q_s.index.tolist()

        highestLevel = max([self.options[opt].level for opt in optsAvailable])
        optsAtLevel = [
            opt for opt in optsAvailable
            if self.options[opt].level == highestLevel
        ]

        if self.talkative:
            print("all opts available are", optsAvailable)
            print("opts at highest level (", highestLevel, ") are",
                  optsAtLevel)

        return(highestLevel, optsAtLevel)

    def move(self, env):
        '''
        Moves the agent according to the lowest level (primitive) currently
        active option.

        Expects     env - an object of class Environment

        '''
        action = self.activeOptions[-1]["label"]

        if self.talkative:
            print("moving", action, "from", env.state["label"])

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
            env.state["coords"] = self.s_origin["coords"][:]
        else:
            env.state["coords"][2] = 0

        env.update()

        # If the new state is a sub-goal, record visitation:
        if env.state["label"] == env.SG:
            self.SG_visited = True

        self.stepCounter += 1

        if self.talkative:
            print("entered", env.state["label"], "at", env.state["coords"])
            print("is origin?", env.state["coords"][:2] == [0, 0])

    def collectReward(self, env):
        '''
        Collects reward from the environment if available. If the task is
        hierarchical, reward at the goal location is contingent upon the agent
        having first visited the sub-goal, and so this must be evaluated prior
        to award.

        Expects        env - an object of class Environment
                 task_mode - a str describing the mode of the current task

        '''
        self.r += env.deliverReward(self.SG_visited)

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

        controlOption = self.options[self.activeOptions[-1]["label"]]

        if controlOption.s_term[env.state["label"]][0] == 1:
            self.terminateOption(env)

    def terminateOption(self, env):
        '''
        Terminates the lowest level currently active option, and stores its
        name for later evaluation.

        '''
        self.terminatedOption = self.activeOptions.pop(-1)

        # If active_policies is empty, we update Q:
        if self.activeOptions == []:
            self.update_Q(env)
        else:
            self.checkForTermination(env)

    def update_Q(self, env):
        '''
        Upon termination of all active policies, evaluate the Q-value
        associated with the highest level and most recently active policy (ie
        the policy that was terminated prior to control being handed to the
        Q-matrix) and the state in which it was initiated.

        Expects        env - an object of class Environment

        '''
        s_prev = self.terminatedOption["stateInitialised"]["label"]
        s_curr = env.state["label"]
        o = self.terminatedOption["label"]
        if not self.representsHistory:
            Q = self.Q
        else:
            Q = self.Q[self.s_origin["label"]]

        if self.talkative:
            print(
                "updating value of", o,
                "initialised at", s_prev,
                "; now at", s_curr, "\n"
            )

        delta = self.alpha * (
            self.r + self.gamma * np.max(Q.loc[:, s_curr]) -
            Q.loc[o, s_prev]
        )

        Q.loc[o, s_prev] += delta

        if s_prev in self.statesForAbstraction:
            s_abs = [
                s for s in self.statesForAbstraction if s is not s_prev
            ][0]

            Q.loc[o, s_abs] = Q.loc[o, s_prev]

        if self.r > 0:
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
