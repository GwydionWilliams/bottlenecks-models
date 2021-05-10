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
        self.betas = simParams["betas"]
        self.etas = simParams["etas"]
        self.phis = simParams["phis"]

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
                "agentClass": None,
                "modelNum": None,
                "repNum": None,
                "alpha": None,
                "beta": None,
                "eta": None,
                "regime": None,
                "trialNum": None,
                "subGoalSide": None,
                "goalSide": None,
                "actionHistory": None,
                "stateHistory": None,
                "numSteps": None
            }
            self.dataPath = simParams["dataPath"]
            # self.initDataFile()

    def simulate(self, agentParams):
        self.modelNum = 0

        for a in self.alphas:
            for b in self.betas:
                for h in self.etas:
                    for p in self.phis:
                        for self.repNum in range(self.numReps):
                            agentParams["alpha"] = a
                            agentParams["beta"] = b
                            agentParams["eta"] = h
                            agentParams["phi"] = p

                            self.run(agentParams)

                        self.modelNum += 1
                        print(f"Completed all reps for model number {self.modelNum}.")

    def fit(self, x, d, agentParams):
        agentParams["alpha"] = x[0]
        agentParams["beta"] = x[1]
        if "hierarchical" in agentParams["class"]:
            agentParams["phi"] = x[2]
            if "sequential" in agentParams["selectionStrategy"]:
                agentParams["eta"] = x[3]

        ll = 0
        for dom in d.domain.unique():
            for set in d.policySet.unique():
                d_s = d[
                    (d.domain == dom) &
                    (d.policySet == set)
                ].reset_index()

                ll += self.run(agentParams, d_s)

        nll = -ll

        print(f"for params {np.round(x, 8)}, nll = {np.round(nll, 8)}")

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

                lik = self.agent.selectOption(self.env, empBehav)
                if fitting:
                    ll += np.log(lik)

                self.agent.move(self.env)
                self.agent.collectReward(self.env)
                self.agent.checkForTermination(self.env)

                if self.writeData:
                    if self.agent.activeOptions == []:
                        self.recordOption()

            if self.writeData:
                self.recordTrial()

            t += 1

            # print(f"completed trial {self.trialNum} in {self.agent.stepCounter} steps")
            # print("\n\n\n\n\n\n")

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
                "REP" if (self.trialNum) < (self.numTrials/2) else "ALT"
        else:
            self.activeRegime = regime

        G_side = SG_side if self.activeRegime == "REP" \
            else [side for side in self.G_sides if side != SG_side][0]

        self.env.placeReward(SG_side, G_side)

        if self.writeData:
            self.data["repNum"] = str(self.repNum)
            self.data["modelNum"] = str(self.modelNum)
            self.data["trialNum"] = str(self.trialNum)
            self.data["alpha"] = str(np.round(self.agent.alpha, 2))
            if self.agent.eta is not "NA":
                self.data["eta"] = str(np.round(self.agent.eta, 2))
            else:
                self.data["eta"] = self.agent.eta
            self.data["beta"] = str(np.round(self.agent.beta, 2))
            self.data["regime"] = self.activeRegime
            self.data["subGoalSide"] = SG_side
            self.data["goalSide"] = G_side
            self.data["actionHistory"] = []
            self.data["stateHistory"] = []

    def recordOption(self):
        self.data["actionHistory"].append(self.agent.terminatedOption["label"])

    def recordState(self):
        self.data["stateHistory"].append(self.env.state["label"])

    def recordTrial(self):
        self.data["actionHistory"] = "-".join(self.data["actionHistory"])
        self.data["stateHistory"] = "-".join(self.data["stateHistory"])
        self.data["numSteps"] = str(self.agent.stepCounter)
        self.data["agentClass"] = self.agent.type

        row = ",".join(list(self.data.values())) + "\n"

        f = open(self.dataPath, "a")
        f.write(row)
        f.close()

    def initDataFile(self):
        headers = ",".join(list(self.data.keys())) + "\n"

        f = open(self.dataPath, "w")
        f.write(headers)
        f.close()
