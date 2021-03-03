from simulation import Simulation
from sim_funs import buildEnv, definePrimitiveActions, defineOptions, writeData
import numpy as np

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
simParams = {
    "numTrials": int(2e3),
    "taskMode": "hierarchical",
    "dataPath": "./data/placeholder"
}

agentParams = {
    "class": "abstract-hierarchical",
    "policy": "softmax",
    "gamma": .5,
    "alpha": np.arange(0.02, 1, step=.02)
}

envParams = {
    "regimes": ["repeat", "alternate"]
}
#
# data_dir = "./data/"
# file_name = "31-MFA0-hierEnv"
#
# #    ii.  AGENT & ENVIRONMENT
# alphas = np.arange(0.02, 1, step=.02)
# gamma = .5
# policy = "softmax"
# options = defineOptions(simParams["agentClass"], simParams["taskMode"])
#
# states = buildEnv(simParams["task_mode"])

for alpha in agentParams["alpha"]:

    # agentParams = {
    #     "alpha": alpha,
    #     "gamma": gamma,
    #     "options": options,
    #     "policy": policy,
    #     "agent_class": simParams["agentClass"]
    # }
    #
    # env_params = {
    #     "states": states
    # }

    #    iv. CONTROLLER
    sim = Simulation(simParams, agentParams, envParams)

    # -------------------------------------------------------------------------
    # 2. RUN SIMULATION -------------------------------------------------------
    for sim.trialNum in range(sim.numTrials):
        # print("_________________________NEW  TRIAL_________________________")

        sim.setupTrial()

        sim.agent.wakeUp()

        while sim.agent.r is None:
            pass

        print("trial completed")

        # while sim.agent.terminationReached is not True:

        # while sim.agent.under_primitive_control is False:
        # sim.agent.select_option(sim.env)

        # sim.agent.move(sim.env)
        # sim.agent.collect_reward(sim.env, sim.task_mode)
        # sim.agent.check_for_termination()
        # if sim.agent.under_Q_control:
        #     sim.agent.update_Q(sim.env)
        # sim.norm_Q()

        # sim.summarise_step()

        # sim.t += 1

        # sim.record_trial()

        if sim.trialNum != 0:
            # if ((sim.n_trial % (sim.num_trials / 10)) == 0) or \
            #         (sim.n_trial == (sim.num_trials-1)):
            #     sim.summarise_chunk()

            if ((sim.trialNum % (sim.numTrials / 2)) == 0):
                sim.switchRegime()

    # -------------------------------------------------------------------------
    # 3. SAVE RESULTS ---------------------------------------------------------
    # write_data(sim, data_dir, file_name)
