import csv
import numpy as np


def build_env(mode):
    if mode is "hierarchical":
        states = [[0, 0, 0], [0, 0, 1],
                  [-1, 1, 0], [1, 1, 0], [-2, 2, 0], [0, 2, 0], [2, 2, 0],
                  [-1, 3, 0], [1, 3, 0]]
        state_labels = ["B0L", "B0R",
                        "SGL", "SGR", "DL", "B1", "DR",
                        "GL", "GR"]

    elif mode is "flat":
        states = [[0, 0, 0], [-1, 1, 0], [1, 1, 0],
                  [-2, 2, 0], [0, 2, 0], [2, 2, 0],
                  [-1, 3, 0], [1, 3, 0]]
        state_labels = ["B0", "SGL", "SGR",
                        "DL", "B1", "DR",
                        "GL", "GR"]

    return states, state_labels


def define_primitive_actions(action_lbls, mode):
    s_initiation = {
        action_lbls[0]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        action_lbls[1]: [0, 0, 1, 0, 1, 1, 0, 1, 0],
        action_lbls[2]: [0, 0, 0, 1, 0, 1, 1, 0, 1],
        action_lbls[3]: [1, 1, 1, 1, 0, 1, 0, 0, 0]
    }

    s_termination = {
        action_lbls[0]: [0, 0, 0, 1, 0, 1, 1, 0, 1],
        action_lbls[1]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        action_lbls[2]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        action_lbls[3]: [0, 0, 1, 0, 1, 1, 0, 1, 0]
    }

    if mode is "flat":
        for a in action_lbls:
            s_initiation[a] = s_initiation[a][1:]
            s_termination[a] = s_termination[a][1:]

    num_actions = len(action_lbls)
    num_states = len(s_initiation[action_lbls[0]])

    pi = {}
    for i, a in enumerate(s_initiation.keys()):
        pi[a] = np.zeros((num_actions, num_states))
        for s in range(num_states):
            if s_initiation[a][s] == 1:
                pi[a][i, s] = 1

    return s_initiation, s_termination, pi


def define_options(agent_class, task_mode):
    if agent_class is "flat":
        labels = s_init = s_term = pi = []
    elif "hierarchical" in agent_class:
        labels = ["B0_B1_L", "B0_B1_R",
                  "B0_GL_REP", "B0_GR_REP",
                  "B0_GL_ALT", "B0_GR_ALT"]

        s_init = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
        ]

        s_term = [
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        ]

        pi = [
            np.array([
                0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((4, 9)),
            np.array([
                1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0,
            ]).reshape((4, 9)),
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
            np.array([
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
            np.array([
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
        ]

        if "abstract" in agent_class:
            for lbl in ["REP", "ALT"]:
                labels.append(lbl)

            for opt in range(2):
                s_init.append(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]))
                s_term.append(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]))

            pi.append(
                np.array([
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]).reshape((10, 9))
            )

            pi.append(
                np.array([
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0,
                ]).reshape((10, 9))
            )

        if task_mode is "flat":
            for o in range(len(labels)):
                s_init[o] = s_init[o][1:]
                s_term[o] = s_term[o][1:]
                pi[o] = pi[o][:, 1:]

    return labels, s_init, s_term, pi


def write_data(sim, dir_name, file_name):
    file_path_csv = dir_name + file_name + ".csv"
    file_path_Q = dir_name + file_name + "_Q.npy"
    print("Writing data to {0}...".format(file_path_csv))
    with open(file_path_csv, 'a') as writeFile:
        if writeFile is "":
            writeFile.write(
                "alpha,num_steps,mu_steps,sub_goal_side,goal_side," +
                "regime,action_history,state_history\n"
            )

        writer = csv.writer(writeFile)

        output = []
        for key, item in sim.data.items():
            output.append(item)

        output = zip(*output)

        writer.writerows(output)

    np.save(file_path_Q, sim.data_Q)


def find_state(state, env, value="index"):
    if value is "index":
        return env.states.index(state)
    elif value is "label":
        return env.state_lbls[env.states.index(state)]
