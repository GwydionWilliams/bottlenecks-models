from simulation import Simulation
from sim_funs import buildEnv, definePrimitiveActions, defineOptions, writeData
import numpy as np

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
simParams = {
    "numTrials": int(1e3),
    "taskMode": "hierarchical",
    "dataPath": "./data/placeholder"
}

agentParams = {
    "class": "abstract-hierarchical",
    "policy": "softmax",
    "tau": .5,
    "beta": .2,
    "gamma": .5,
    "alpha": .5
}

envParams = {
    "regimes": ["repeat", "alternate"]
}

sim = Simulation(simParams, agentParams, envParams)

# -------------------------------------------------------------------------
# 2. RUN SIMULATION -------------------------------------------------------
for sim.trialNum in range(sim.numTrials):

    sim.setupTrial()

    sim.agent.wakeUp(sim.env)

    while sim.agent.awake:
        sim.agent.selectOption(sim.env)
        sim.agent.move(sim.env)
        sim.agent.collectReward(sim.env)
        sim.agent.checkForTermination(sim.env)

    print("trial ", sim.trialNum, " under ", sim.activeRegime,
          " completed in ", sim.agent.stepCounter, " steps")

    if ((sim.trialNum % (sim.numTrials / 2)) == 0 and sim.trialNum != 0):
        print(sim.agent.Q)
        sim.switchRegime()

print(sim.agent.Q)

# -------------------------------------------------------------------------
# 3. SAVE RESULTS ---------------------------------------------------------
# write_data(sim, data_dir, file_name)
