import copy


class Environment():
    def __init__(self, params):
        self.T = None

        self.states = params["states"]
        self.state = None

        self.pR = {}
        for s in self.states.keys():
            self.pR[s] = 0

        self.SG = None
        self.G = None

    def update(self):
        newState = self.findState()
        self.state["label"] = newState

    def placeReward(self, G_side, SG_side):
        s_origin = "B0" + SG_side
        self.state = {
            "label": s_origin,
            "coords": copy.deepcopy(self.states[s_origin])
        }

        self.G = "G" + G_side
        self.SG = "SG" + SG_side

        self.pR["G" + G_side] = 1

    def findState(self):
        print(self.state)
        for stateLabel, stateCoords in self.states.items():
            if self.state["coords"] == stateCoords:
                return(stateLabel)

    def deliverReward(self, SG_visited):
        if SG_visited:
            r = self.pR[self.state["label"]]
        else:
            r = 0

        return(r)
