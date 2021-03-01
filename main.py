from simulation import Simulation
from sim_funs import build_env, define_options, write_data
import numpy as np

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
sim_params = {
    "num_trials": int(2e3),
    "task_mode": "hierarchical",
    "agent_class": "abstract-hierarchical",
    "agent_represents_history": False,
    "regime": ["repeat", "alternate"]
}

data_dir = "./data/"
file_name = "31-MFA0-hierEnv"

#    ii.  AGENT & ENVIRONMENT
action_lbls = ["NE", "SE", "SW", "NW"]
alphas = np.arange(0.02, 1, step=.02)
gamma = .5
policy = "e-greedy"
epsilon = {
    "start": 0.2,
    "end": 0.01,
    "decay": 100
}

for alpha in alphas:
    print("          RUNNING SIM FOR {0}".format(alpha))
    states, state_labels = build_env(sim_params["task_mode"])
    labels, s_init, s_term, pi = define_options(
        sim_params["agent_class"], sim_params["task_mode"]
    )

    agent_params = {
        "alpha": alpha,
        "gamma": gamma,
        "action_lbls": action_lbls,
        "policy": policy,
        "epsilon": epsilon,
        "agent_class": sim_params["agent_class"],
        "has_history": sim_params["agent_represents_history"]
    }

    env_params = {
        "states": states,
        "state_labels": state_labels
    }

    option_params = {
        "label": labels,
        "s_init": s_init,
        "s_term": s_term,
        "pi": pi,
    }

    #    iv. CONTROLLER
    sim = Simulation(agent_params, env_params, option_params, sim_params)

    # -------------------------------------------------------------------------
    # 2. RUN SIMULATION -------------------------------------------------------
    for sim.n_trial in range(sim.num_trials):
        # print("_________________________NEW  TRIAL_________________________")

        sim.setup_trial()
        sim.t = 0

        while sim.agent.termination_reached is not True:

            while sim.agent.under_primitive_control is False:
                sim.agent.select_option(sim.env)

            sim.agent.move(sim.env)
            sim.agent.collect_reward(sim.env, sim.task_mode)
            sim.agent.check_for_termination()
            if sim.agent.under_Q_control:
                sim.agent.update_Q(sim.env)
                # sim.norm_Q()

            # sim.summarise_step()

            sim.t += 1

        sim.record_trial()

        if sim.n_trial != 0:
            if ((sim.n_trial % (sim.num_trials / 10)) == 0) or \
                    (sim.n_trial == (sim.num_trials-1)):
                sim.summarise_chunk()

            if ((sim.n_trial % (sim.num_trials / 2)) == 0):
                sim.switch_regime()

    # -------------------------------------------------------------------------
    # 3. SAVE RESULTS ---------------------------------------------------------
    write_data(sim, data_dir, file_name)
