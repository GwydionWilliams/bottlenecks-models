import numpy as np
import pandas as pd
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


def softmax(Q, beta):
    """Compute softmax values for each sets of scores in x."""
    e_Q = np.exp((Q/beta).astype(float))
    return (e_Q / e_Q.sum(axis=0)).fillna(1)


def findMatchingBehaviour(agent, env, empiricalBehaviour):
    if agent.talkative:
        print(f"empirical actions left this trial: " +
              "{empiricalBehaviour['actions']}")

    hierarchy = findLegalHierarchy(agent, env, empiricalBehaviour)

    if hierarchy is None:
        hierarchy = [empiricalBehaviour["actions"][0]]

    return(hierarchy)


def findLegalHierarchy(agent, env, empiricalBehaviour):
    optLevels = np.unique(
        [o.level for o in agent.options.values()]
    )
    h, H = 1, max(optLevels)

    # 1. Find legal hierarchies at t0 -------------------------------------
    hierarchies = [[empiricalBehaviour["actions"][0]]]
    while h <= H:
        for i in range(len(hierarchies)):
            options = findLegalParentOpts(agent, hierarchies[i][0], h, env)

            if len(options) > 1:
                for j in range(len(options)-1):
                    hierarchies.append(copy.deepcopy(hierarchies[j]))

                for k in range(len(options)):
                    hierarchies[k].insert(0, options[k])

            elif len(options) == 1:
                hierarchies[i].insert(0, options[0])

        h += 1

    # 2. Check validity of legal hierarchies for remainder of sequence ----
    if len(hierarchies) is 1 and len(hierarchies[0]) is 1:
        hierarchy = None
    else:
        validHierarchies = checkHierarchyValidity(
            agent, env, hierarchies, empiricalBehaviour
        )

        hierarchy = validHierarchies[0] if validHierarchies != [] else None

    return(hierarchy)


def findLegalParentOpts(agent, option, h, env):
    Q_s = agent.Q.loc[:, env.state["label"]]
    optsAvailable = Q_s.dropna().index.tolist()

    optsAtLevel = [
        opt for opt in optsAvailable
        if agent.options[opt].level == h
    ]

    legalOpts = [
        opt for opt in optsAtLevel
        if agent.options[opt].pi.loc[option, env.state["label"]] is not None
    ]

    return(legalOpts)


def checkHierarchyValidity(agent, env, hierarchies, empiricalBehaviour):
    candidateHierarchies = copy.deepcopy(hierarchies)
    invalidHierarchies = []
    for i, hierarchy in enumerate(candidateHierarchies):
        hierarchy.pop(-1)
        hierarchyInvalid = False

        for a, s in zip(empiricalBehaviour["actions"][1:],
                        empiricalBehaviour["states"][1:]):

            # i. Correct neutral origin states
            s = env.origin["label"] if s == "B0" else s

            # ii. Check for option termination
            if agent.options[hierarchy[-1]].s_term.loc[:, s][0] is 1:
                hierarchy.pop(-1)

                if hierarchy == []:
                    break

            # iii. Check if prescribed primitve matches empirical primitive
            newOption, newOptionLbl, primitive = None, None, None
            while not isinstance(newOption, PrimitiveAction):
                if newOption is not None:
                    hierarchy.append(newOptionLbl)

                currentPi = agent.options[hierarchy[-1]].pi.loc[:, s]

                try:
                    newOptionLbl = currentPi.dropna().index[0]
                    newOption = agent.options[newOptionLbl]

                except IndexError:
                    hierarchyInvalid = True
                    break
            else:
                primitive = newOptionLbl
                hierarchyInvalid = not primitive == a

            if hierarchyInvalid:
                invalidHierarchies.append(i)
                break

    for i in invalidHierarchies[::-1]:
        hierarchies.pop(i)

    return(hierarchies)


def fetchChoiceProbability(agent):
    choice = agent.activeOptions[0]
    choiceLevel = agent.options[choice["label"]].level
    highestLevel = max([option.level for option in agent.options.values()])

    if agent.representsHistory:
        Q = agent.Q[agent.s_origin["label"]]
        beta = agent.beta
    else:
        Q = agent.Q
        if "sequential" in agent.selectionStrategy:
            beta = agent.betas[choiceLevel]
        else:
            beta = agent.beta

    optsAtLevel = [
        opt for opt in agent.options
        if agent.options[opt].level == choiceLevel
    ]

    Q = Q.loc[optsAtLevel, choice["stateInitialised"]["label"]].dropna()
    # print(f"choice was {choice['label']} (level {choiceLevel})")
    # print(
    #     f"was this (level {choiceLevel}) highest level (level {highestLevel})? {choiceLevel == highestLevel}")

    Q = softmax(Q, beta)

    p = Q.loc[choice["label"]]
    if "sequential" in agent.selectionStrategy:
        forgotHierarchy = choiceLevel != highestLevel

        if forgotHierarchy:
            p *= agent.phi
        else:
            p *= (1 - agent.phi)

    if (choiceLevel != highestLevel) and ("sequential" in agent.selectionStrategy):
        # print("forgot therefore reset")
        p = 1
    #
    # print(f"choice taken with prob {p}, or log(p) is {np.log(p)}")
    # print(
    #     f"with prob(forgetting), p becomes {p * (1-agent.phi)}, and log(p) becomes {np.log(p * (1-agent.phi))}")

    return(p)

# -----------------------------------------------------------------------------
# ----------------------------------- AGENT -----------------------------------
# -----------------------------------------------------------------------------


class Agent():
    def __init__(self, params, env):
        self.type = params["class"]
        self.awake = False
        self.talkative = params["talkative"]

        self.s_origin = None
        self.representsHistory = params["representsHistory"]
        self.selectionStrategy = params["selectionStrategy"]
        self.statesForAbstraction = params["statesForAbstraction"]

        self.options = params["options"]

        self.r = None

        self.alpha = params["alpha"]  # learning rate
        self.gamma = params["gamma"]  # discounting factor
        self.phi = params["phi"]  # p(forget hierarchy)

        self.selectionPolicy = params["policy"]
        if self.selectionPolicy == "e-greedy":
            self.eps = params["epsilon"]
        elif "softmax" in self.selectionPolicy:
            self.beta = params["beta"]

            if "structured" in self.selectionPolicy:
                self.eta = params["eta"]

                optLevels = np.unique(
                    [o.level for o in self.options.values()]
                )

                self.betas = self.beta * \
                    self.eta ** (max(optLevels) - optLevels)

            else:
                self.eta = "NA"

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

        if self.talkative:
            print("\n\nstarting trial at", self.s_origin["label"], env.origin["coords"])

    def selectOption(self, env, empiricalBehaviour=None):
        '''
        Selects an option from the policy given by the highest level currently
        active option. If no options are currently active, one will be selected
        from the agent's Q matrix.

        Expects     env - an object of class Environment

        '''
        # ------------------------------ FITTING ------------------------------
        behaviourGiven = empiricalBehaviour is not None

        if behaviourGiven:
            if self.activeOptions == []:
                hierarchy = \
                    findMatchingBehaviour(self, env, empiricalBehaviour)

                for option in hierarchy:
                    self.activeOptions.append(
                        {"label": option,
                         "stateInitialised": copy.deepcopy(env.state)}
                    )

            else:
                self.findPrescribedOption(env)

            for record in empiricalBehaviour.values():
                record.pop(0)

            p = fetchChoiceProbability(self)

            # print(f"selected option {self.activeOptions[0]['label']} with ll {np.log(p)}")
            # if forgotHierarchy:
            #     print(
            #         f"forgot hierarchy at {env.state['label']} with option {self.activeOptions[0]['label']}")
            # else:
            #     print(
            #         f"did not forgot hierarchy at {env.state['label']} with option {self.activeOptions[0]['label']}")
            return(p)

        # ---------------------------- SIMULATION -----------------------------
        # 0. Forget hierarchy -------------------------------------------------
        if "sequential" in self.selectionStrategy:
            p = np.random.uniform(low=0, high=1)
            forgetHierarchy = p < self.phi
            if forgetHierarchy:
                self.activeOptions = []

                _, opts = self.findOptionsAtLevel(env, level=0)
                Q = self.Q.loc[opts, env.state["label"]]

                choice = self.selectFromQ(Q, 0)

                self.activeOptions.append(
                    {"label": choice,
                     "stateInitialised": copy.deepcopy(env.state)}
                )

                if self.talkative:
                    print("oops, forgot hierarchy!")

                return()

        # 1. Find Q_sa --------------------------------------------------------
        if self.activeOptions == []:

            # 1a. Find Q_s for current s (if selection strategy is sequential,
            #     restrict option choice to highest level possible)
            if not self.representsHistory:
                pi = self.Q
            else:
                pi = self.Q[self.s_origin["label"]]

            if self.selectionStrategy is "sequential":
                level, opts = self.findOptionsAtLevel(env, level="max")
                pi = pi.loc[opts, ]
            else:
                level = "all"

            Q_sa = pi.loc[:, env.state["label"]]

            # 2. Select action from Q_sa --------------------------------------
            # 2a. greedy selection
            choice = self.selectFromQ(Q_sa, level)

            self.activeOptions.append(
                {"label": choice, "stateInitialised": copy.deepcopy(env.state)}
            )

        else:
            # Ad. If there is an active high-level option, recover choice from
            #     memory
            choice = self.activeOptions[-1]["label"]

        # 3. Check if lowest-level option is primitive ------------------------
        if isinstance(self.options[choice], PrimitiveAction) is False:
            self.findPrescribedOption(env)
        else:
            if self.talkative:
                print(
                    "selected action hierarchy is: ",
                    [o["label"] for o in self.activeOptions]
                )

    def selectFromQ(self, Q, level="all"):
        if self.selectionPolicy == "greedy":
            Q_max = Q.transform(lambda x: x == x.max()).astype('bool')
            maxRow = Q.loc[Q_max]
            if len(maxRow) > 1:
                maxRow = maxRow.sample()
            choice = maxRow.index[0]

        # 2b. e-greedy selection
        elif self.selectionPolicy == "e-greedy":
            if np.random.random() < self.eps:
                choice = np.random.choice(np.where(Q > 0)[0])

        # 2cd. structured-/softmax selection
        elif "softmax" in self.selectionPolicy:
            if "structured" in self.selectionPolicy:
                beta = self.betas[level]

                if self.talkative:
                    print("selecting option from level", level,
                          "with temperature", beta)
            else:
                beta = self.beta

            Q = Q.dropna()
            Q = softmax(Q, beta)
            if self.talkative:
                print(f"selecting from: \n{Q}")

            choice = Q.sample(weights=Q).index[0]

            return(choice)

    def findPrescribedOption(self, env):
        pi = self.options[self.activeOptions[-1]["label"]].pi
        pi_s = pi.loc[:, env.state["label"]]

        pi_max = pi_s.transform(lambda x: x == x.max()).astype('bool')
        maxRow = pi_s.loc[pi_max]
        if len(maxRow) > 1:
            maxRow = maxRow.sample()

        choice = maxRow.index[0]

        self.activeOptions.append(
            {"label": choice,
             "stateInitialised": copy.deepcopy(env.state)}
        )

        if isinstance(self.options[choice], PrimitiveAction) is False:
            self.findPrescribedOption(env)
        else:
            if self.talkative:
                print(f"selected hierarchy is: {[opt['label'] for opt in self.activeOptions]}")

    def findOptionsAtLevel(self, env, level="max", state=None):
        if state is None:
            state = env.state["label"]

        Q_s = self.Q.loc[:, state]
        Q_s = Q_s.dropna()
        optsAvailable = Q_s.index.tolist()

        if level == "max":
            level = max([self.options[opt].level for opt in optsAvailable])

        optsAtLevel = [
            opt for opt in optsAvailable
            if self.options[opt].level == level
        ]

        # if self.talkative:
        #     print("all opts available are", optsAvailable)
        #     print("opts at level (", level, ") are",
        #           optsAtLevel)

        return(level, optsAtLevel)

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
        env.checkElevation(action)
        env.update()

        # If the new state is a sub-goal, record visitation:
        if env.state["label"] == env.SG:
            self.SG_visited = True

        self.stepCounter += 1

        if self.talkative:
            print(f"entered {env.state['label']} at {env.state['coords']}; " +
                  f"SG visited? {self.SG_visited}")
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

        if self.talkative:
            print(f"terminating {self.terminatedOption['label']}\n")

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

        delta = self.alpha * (
            self.r + self.gamma * np.max(Q.loc[:, s_curr]) -
            Q.loc[o, s_prev]
        )

        Q.loc[o, s_prev] += delta
        Q.loc[o, s_prev] = np.clip(Q.loc[o, s_prev], 0, 1)

        if s_prev in self.statesForAbstraction:
            s_abs = [
                s for s in self.statesForAbstraction if s != s_prev
            ][0]

            print(f"abstracting learning from {s_prev} to {s_abs}")

            Q.loc[o, s_abs] = Q.loc[o, s_prev]

        if self.talkative:
            if not self.representsHistory:
                print(
                    f"updating value of {o} initialised at {s_prev}; now at {s_curr} with Q: \n{self.Q}\n"
                )
            else:
                print(
                    f"updating value of {o} initialised at {s_prev}; now at {s_curr} with Qs: Q_B0L:\n{self.Q['B0L']}\nQ_B0R:\n{self.Q['B0R']}"
                )

            if env.state["label"] == self.s_origin["label"]:
                print(f"back at origin {self.s_origin['label']}; (coords: {env.state['coords']})")

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
