import numpy as np


class Environment():
    def __init__(self, states, state_lbls):
        self.T = None
        self.states = states
        self.state_lbls = state_lbls
        self.num_states = len(states)
        self.state_i = np.arange(self.num_states)

    def init_pR(self):
        self.pR = np.zeros(self.num_states)

    def place_r(self, G_side, SG_side, SG_i):
        self.pR[self.state_lbls.index("G" + G_side)] = 1
        self.SG = "SG" + SG_side
