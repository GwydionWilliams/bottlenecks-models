import copy


class Environment():
    def __init__(self, params):
        self.T = None

        self.states = params["states"]
        self.state = None

        self.SG = None
        self.G = None

    def update(self):
        newState = self.findState()
        self.state["label"] = newState

    def placeReward(self, SG_side, G_side):
        self.pR = {}
        for s in self.states.keys():
            self.pR[s] = 0

        originLbl = "B0" + SG_side
        self.origin = {
            "label": originLbl,
            "coords": copy.deepcopy(self.states[originLbl])
        }

        self.state = copy.deepcopy(self.origin)

        self.SG = "SG" + SG_side
        self.G = "G" + G_side

        self.pR["G" + G_side] = 1

    def checkElevation(self, action):
        if self.state["label"] == "VL" and action == "NE":
            self.state["coords"][2] = 1
        elif self.state["coords"][0:2] == self.origin["coords"][0:2]:
            self.state["coords"][2] = 1
        else:
            self.state["coords"][2] = 0

    def checkForOrigin(self, origin):
        if self.state["coords"][0:2] == origin["coords"][0:2]:
            self.state["coords"][2] = origin["coords"][0:2]
        else:
            self.state["coords"][2] = 0

    def findState(self):
        for stateLabel, stateCoords in self.states.items():
            if self.state["coords"] == stateCoords:
                return(stateLabel)

    def deliverReward(self, SG_visited):
        if SG_visited:
            r = self.pR[self.state["label"]]
        else:
            r = 0

        return(r)
