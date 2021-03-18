import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
import agentClasses
from simulation import Simulation

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
simParams = {
    "mode": "fit",
    "numTrials": 200,
    "alphas": np.arange(.1, 1, .1),
    "taus": np.arange(.1, 2, .2),
    "numReps": 10,
    "writeData": False
}

agentParams = agentClasses.params["abstract-hierarchical"]
agentParams.update({
    "beta": .8,
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

# -----------------------------------------------------------------------------
# 2. RUN SIMULATION -----------------------------------------------------------
sim = Simulation(simParams, envParams)

if sim.mode == "simulate":
    sim.simulate(agentParams)
elif sim.mode == "fit":
    d = pd.read_csv("./data/EmpiricalData.csv")
    d = d.loc[
        :, ["subjNum", "policySet", "episode", "domain", "policy", "path",
            "statesVisited", "actions", "steps"]
    ]

    for subj in d.subjNum.unique():
        print(f" - - - - - SUBJ {subj} | {agentParams['class']} - - - - - ")

        d_s = d[d.subjNum == subj].reset_index()
        bounds = Bounds([0.01, 0.01], [1, 5])
        x0 = np.array([.9, .2])
        sol = minimize(
            sim.fit, x0, args=(d_s, agentParams),
            bounds=bounds
        )

        print(sol)

        # print(sim.fit([.1, .5], d, agentParams))
