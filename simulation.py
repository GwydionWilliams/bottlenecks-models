import numpy as np
from agent import Agent, softmax
from environment import Environment
from auxFuns import buildEnv, defineOptions


class Simulation():
    def __init__(self, simParams, envParams):
        self.mode = simParams["mode"]

        self.numReps = simParams["numReps"]
        self.writeData = simParams["writeData"]

        self.alphas = simParams["alphas"]
        self.taus = simParams["taus"]

        self.numTrials = simParams["numTrials"]
        self.regimes = envParams["regimes"]
        self.activeRegime = self.regimes[0]

        states = buildEnv()
        envParams.update({"states": states})

        self.env = Environment(envParams)

        self.SG_sides = ["L", "R"]
        self.G_sides = ["L", "R"]

        if self.writeData:
            self.data = {
                "modelNum": None,
                "repNum": None,
                "alpha": None,
                "tau": None,
                "beta": None,
                "regime": None,
                "trialNum": None,
                "subGoalSide": None,
                "goalSide": None,
                "actionHistory": None,
                "stateHistory": None,
                "numSteps": None
            }
            self.dataPath = simParams["dataPath"]
            self.initDataFile()

    def simulate(self, agentParams):
        self.modelNum = 0

        for a in self.alphas:
            for t in self.taus:
                for self.repNum in range(self.numReps):
                    self.run(agentParams, a, t)
                    print(self.agent.Q)

                self.modelNum += 1
                print(f"Completed all reps for model number {self.modelNum}.")

    def fit(self, x, d, agentParams):
        agentParams["alpha"] = x[0]
        agentParams["tau"] = x[1]

        ll = 0

        for dom in d.domain.unique():
            for set in d.policySet.unique():
                d_s = d[
                    (d.domain == dom) &
                    (d.policySet == set)
                ].reset_index()

                ll += self.run(agentParams, d_s)

        nll = -ll

        print(f"for params {x}, nll = {nll}")

        return(-ll)

    def run(self, agentParams, d=None):
        if d is None:
            fitting = False
            numTrials = self.numTrials
            path, regime, empBehav = None, None, None
        else:
            numTrials = d.shape[0]
            fitting = True
            ll = 0

        self.setupAgent(agentParams)

        for self.trialNum in range(numTrials):
            if fitting:
                path = d.loc[self.trialNum, ].path.upper()
                regime = d.loc[self.trialNum, ].policy.upper()
                empBehav = {
                    "actions": d.loc[self.trialNum, ].actions.split("-"),
                    "states": d.loc[self.trialNum, ].statesVisited.split("-")
                }

            self.setupTrial(path=path, regime=regime)

            self.agent.wakeUp(self.env)

            t = 0
            while self.agent.awake:
                if self.writeData:
                    self.recordState()

                self.agent.selectOption(self.env, empBehav)
                self.agent.move(self.env)

                if fitting:
                    ll += self.fetchChoiceProbabilities()

                self.agent.collectReward(self.env)
                self.agent.checkForTermination(self.env)

                if self.writeData:
                    if self.agent.activeOptions == []:
                        self.recordOption()

            if self.writeData:
                self.recordTrial()

            t += 1

            # print("completed trial", self.trialNum,
            #       "in", self.agent.stepCounter, "steps")

        if fitting:
            return(ll)

    def setupAgent(self, agentParams):
        options = defineOptions(
            agentParams["class"],
            self.env.states.keys()
        )

        agentParams.update({"options": options})

        self.agent = Agent(agentParams, self.env)

    def setupTrial(self, regime=None, path=None):
        self.agent.sleep()

        if path is None:
            SG_side = self.SG_sides[self.trialNum % 2]
        else:
            SG_side = path

        if regime is None:
            self.activeRegime = \
                "REP" if (self.trialNum) < (self.numTrials/2) \
                else "ALT"
        else:
            self.activeRegime = regime

        G_side = SG_side if self.activeRegime == "REP" \
            else [side for side in self.G_sides if side != SG_side][0]

        self.env.placeReward(SG_side, G_side)

        # if self.writeData:
        #     self.data["repNum"] = str(self.repNum)
        #     self.data["modelNum"] = str(self.modelNum)
        #     self.data["trialNum"] = str(self.trialNum)
        #     self.data["alpha"] = str(np.round(self.agent.alpha, 2))
        #     if self.agent.beta is not "NA":
        #         self.data["beta"] = str(np.round(self.agent.beta, 2))
        #     else:
        #         self.data["beta"] = self.agent.beta
        #     self.data["tau"] = str(np.round(self.agent.tau, 2))
        #     self.data["regime"] = self.activeRegime
        #     self.data["subGoalSide"] = SG_side
        #     self.data["goalSide"] = G_side
        #     self.data["actionHistory"] = []
        #     self.data["stateHistory"] = []

    def fetchChoiceProbabilities(self):
        stateInitialised = \
            self.agent.activeOptions[0]["stateInitialised"]["label"]

        if self.agent.representsHistory:
            Q = self.agent.Q[self.agent.s_origin["label"]]
            level, options = 0, list(self.agent.options.keys())
            tau = self.agent.tau

        else:
            Q = self.agent.Q
            level, options = \
                self.agent.findHighestLevelOptions(self.env, stateInitialised)
            tau = self.agent.taus[level]

        controlOption = self.agent.activeOptions[0]["label"]

        Q = Q.loc[options, stateInitialised].dropna()
        Q = softmax(Q, tau)

        try:
            choiceProbability = Q.loc[controlOption, ]
        except KeyError:
            choiceProbability = 0.05 * 1/3  # FIX THIS

        # print(f"probability of selecting {controlOption} from " +
        #       f"{stateInitialised} was {choiceProbability}; " +
        #       f"(log(p) = {np.log(choiceProbability)})")

        return(np.log(choiceProbability))

    def recordOption(self):
        self.data["actionHistory"].append(self.agent.terminatedOption["label"])

    def recordState(self):
        self.data["stateHistory"].append(self.env.state["label"])

    def recordTrial(self):
        self.data["actionHistory"] = "-".join(self.data["actionHistory"])
        self.data["stateHistory"] = "-".join(self.data["stateHistory"])
        self.data["numSteps"] = str(self.agent.stepCounter)

        row = ",".join(list(self.data.values())) + "\n"

        f = open(self.dataPath, "a")
        f.write(row)
        f.close()

    def initDataFile(self):
        headers = ",".join(list(self.data.keys())) + "\n"

        f = open(self.dataPath, "w")
        f.write(headers)
        f.close()
