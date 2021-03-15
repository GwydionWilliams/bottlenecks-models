import numpy as np
from simulation import Simulation

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
agentParams = {
    "class": "flat",
    "representsHistory": True,
    "statesForAbstraction": [],
    "selectionStrategy": "free",
    "policy": "softmax",
    "beta": .2,
    "gamma": .5,
    "alpha": None,
    "tau": None,
    "talkative": False
}


envParams = {
    "regimes": ["repeat", "alternate"]
}


simParams = {
    "numTrials": 200,
    "taskMode": "hierarchical",
    "alphas": np.arange(.7, 1, .1),
    "taus": np.arange(.1, 1, .1),
    "numReps": 10
}

fileName = "".join(
    [classParam[0] for classParam in agentParams["class"].split("-")]
).upper() + "-" + ["H" if agentParams["representsHistory"] else "0"][0]
fileName += f'-{simParams["taskMode"][0:4]}Env.csv'

simParams.update({"dataPath": f"./data/{fileName}"})

# -------------------------------------------------------------------------
# 2. RUN SIMULATION -------------------------------------------------------
sim = Simulation(simParams, envParams)
sim.modelNum = 0


for a in simParams["alphas"]:
    for t in simParams["taus"]:
        for sim.repNum in range(sim.numReps):
            sim.reset()

            agentParams["alpha"], agentParams["tau"] = a, t
            sim.setupAgent(agentParams)

            for sim.trialNum in range(sim.numTrials):

                sim.setupTrial()

                sim.agent.wakeUp(sim.env)

                while sim.agent.awake:
                    sim.recordState()
                    sim.agent.selectOption(sim.env)

                    sim.agent.move(sim.env)

                    sim.agent.collectReward(sim.env)

                    sim.agent.checkForTermination(sim.env)
                    if sim.agent.activeOptions == []:
                        sim.recordOption()

                # sim.recordTrial()

                print("trial", sim.trialNum, "under", sim.activeRegime,
                      "completed in", sim.agent.stepCounter, "steps")

                if ((sim.trialNum+1) % (sim.numTrials / 2)) == 0 and \
                        sim.trialNum != 0:
                    if sim.agent.representsHistory:
                        for origin, Q in sim.agent.Q.items():
                            print(Q)
                    else:
                        print(sim.agent.Q)
                    sim.switchRegime()

        sim.modelNum += 1
        print(f"Completed all reps for model number {sim.modelNum}.")

# -------------------------------------------------------------------------
# 3. SAVE RESULTS ---------------------------------------------------------
# write_data(sim, data_dir, file_name)
