import numpy as np
import pandas as pd
import agentClasses
from simulation import Simulation

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
simParams = {
    "mode": "simulate",
    "numTrials": 200,
    "alphas": np.arange(.1, 1, .1),
    "taus": np.arange(.1, 2, .2),
    "numReps": 10,
    "writeData": False
}

agentParams = agentClasses.params["abstract-hierarchical"]
agentParams.update({
    "beta": .2,
    "gamma": .5,
    "alpha": None,
    "tau": None,
    "talkative": False
})

envParams = {
    "regimes": ["REP", "ALT"]
}

fileName = "".join(
    [classParam[0] for classParam in agentParams["class"].split("-")]
).upper() + "-" + ["H" if agentParams["representsHistory"] else "0"][0]
fileName += '-.csv'

simParams.update({"dataPath": f"./data/{fileName}"})

# -------------------------------------------------------------------------
# 2. RUN SIMULATION -------------------------------------------------------
sim = Simulation(simParams, envParams)

if sim.mode == "simulate":
    sim.simulate(agentParams)
elif sim.mode == "fit":
    d = pd.read_csv("./data/EmpiricalData.csv")
    sim.fit(d, agentParams)
