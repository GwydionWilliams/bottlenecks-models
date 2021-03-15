import numpy as np
from agent import Agent
from environment import Environment
from sim_funs import buildEnv, defineOptions


class Simulation():
    def __init__(self, simParams, envParams):
        self.numReps = simParams["numReps"]

        self.taskMode = simParams["taskMode"]
        self.numTrials = simParams["numTrials"]
        self.regimes = envParams["regimes"]
        self.activeRegime = self.regimes[0]

        states = buildEnv(self.taskMode)
        envParams.update({"states": states})

        self.env = Environment(envParams)

        self.SG_sides = ["L", "R"]
        self.G_sides = ["L", "R"]

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

    def setupAgent(self, agentParams):
        options = defineOptions(
            agentParams["class"],
            self.taskMode,
            self.env.states.keys()
        )

        agentParams.update({"options": options})

        self.agent = Agent(agentParams, self.env)

    def reset(self):
        self.activeRegime = self.regimes[0]
        # self.agent.step_counter = 0

    def setupTrial(self):
        self.agent.sleep()

        if self.taskMode is "hierarchical":
            self.G_side = self.SG_sides[self.trialNum % 2]
            if self.activeRegime is "repeat":
                self.SG_side = self.G_side
            elif self.activeRegime is "alternate":
                self.SG_side = [
                    side for side in self.G_sides if side != self.G_side
                ][0]
        else:
            self.G_side = self.SG_sides[self.trialNum % 2]
            if self.G_side.contains("L"):
                self.SG_side = "L"
            else:
                self.SG_side = "R"

        self.env.placeReward(self.G_side, self.SG_side)

        self.t = 0

        self.data["repNum"] = str(self.repNum)
        self.data["modelNum"] = str(self.modelNum)
        self.data["trialNum"] = str(self.trialNum)
        self.data["alpha"] = str(np.round(self.agent.alpha, 2))
        if self.agent.beta is not "NA":
            self.data["beta"] = str(np.round(self.agent.beta, 2))
        else:
            self.data["beta"] = self.agent.beta
        self.data["tau"] = str(np.round(self.agent.tau, 2))
        self.data["regime"] = self.activeRegime
        self.data["subGoalSide"] = self.SG_side
        self.data["goalSide"] = self.G_side
        self.data["actionHistory"] = []
        self.data["stateHistory"] = []

    def recordOption(self):
        self.data["actionHistory"].append(self.agent.terminatedOption["label"])

    def recordState(self):
        self.data["stateHistory"].append(self.env.state["label"])

    def switchRegime(self):
        self.activeRegime = self.regimes[1]
        self.agent.step_counter = 0
        # print("-------------------- REGIME SWITCH --------------------")

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
