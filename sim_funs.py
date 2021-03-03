import csv
import numpy as np
import pandas as pd
from agent import Option, PrimitiveAction


def buildEnv(taskMode):
    if taskMode is "hierarchical":
        states = {
            "B0L": [0, 0, 0],
            "B0R": [0, 0, 1]
        }

    elif taskMode is "flat":
        states = {
            "B0": [0, 0, 0]
        }

    states.update({
        "SGL": [-1, 1, 0],
        "SGR": [1, 1, 0],
        "DL":  [-2, 2, 0],
        "B1":  [0, 2, 0],
        "DR":  [2, 2, 0],
        "GL":  [-1, 3, 0],
        "GR":  [1, 3, 0],
    })

    return states


def definePrimitiveActions(taskMode, states):
    actionLbls = ["NE", "SE", "SW", "NW"]

    s_init = {
        actionLbls[0]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        actionLbls[1]: [0, 0, 1, 0, 1, 1, 0, 1, 0],
        actionLbls[2]: [0, 0, 0, 1, 0, 1, 1, 0, 1],
        actionLbls[3]: [1, 1, 1, 1, 0, 1, 0, 0, 0]
    }

    s_term = {
        actionLbls[0]: [0, 0, 0, 1, 0, 1, 1, 0, 1],
        actionLbls[1]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        actionLbls[2]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        actionLbls[3]: [0, 0, 1, 0, 1, 1, 0, 1, 0]
    }

    if taskMode is "flat":
        for a in actionLbls:
            s_init[a] = s_init[a][1:]
            s_term[a] = s_term[a][1:]

    numActions = len(actionLbls)
    numStates = len(s_init[actionLbls[0]])

    pi = {}
    for i, a in enumerate(actionLbls):
        pi[a] = np.zeros((numActions, numStates))
        for s in range(numStates):
            if s_init[a][s] == 1:
                pi[a][i, s] = 1

    primitiveActions = {}
    for a in actionLbls:
        params = {
            "s_init": pd.DataFrame(s_init[a], index=states).transpose(),
            "s_term": pd.DataFrame(s_term[a], index=states).transpose(),
            "pi": pd.DataFrame(pi[a], columns=states, index=actionLbls)
        }

        primitiveActions[a] = PrimitiveAction(params)

    return primitiveActions


def defineOptions(agentClass, taskMode, states):
    options = definePrimitiveActions(taskMode, states)

    if agentClass is "flat":
        optionLbls = s_init = s_term = pi = []

    elif "hierarchical" in agentClass:
        optionLbls = ["B0_B1_L", "B0_B1_R",
                      "B0_GL_REP", "B0_GR_REP",
                      "B0_GL_ALT", "B0_GR_ALT"]

        allOptions = list(options.keys()) + optionLbls

        s_init = {
            optionLbls[0]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            optionLbls[1]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            optionLbls[2]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            optionLbls[3]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            optionLbls[4]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            optionLbls[5]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
        }

        s_term = {
            optionLbls[0]: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
            optionLbls[1]: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
            optionLbls[2]: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
            optionLbls[3]: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
            optionLbls[4]: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
            optionLbls[5]: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        }

        pi = {
            # B0_B1_L
            optionLbls[0]: np.array([
                0, 0, 1, 0, 0, 0, 0, 0, 0,  # NE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                1, 1, 0, 0, 0, 0, 0, 0, 0,  # NW
            ]).reshape((4, 9)),

            # B0_B1_R
            optionLbls[1]: np.array([
                1, 1, 0, 0, 0, 0, 0, 0, 0,  # NE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                0, 0, 0, 1, 0, 0, 0, 0, 0,  # NW
            ]).reshape((4, 9)),

            # B0_GL_REP
            optionLbls[2]: np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # NE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                0, 0, 0, 0, 0, 1, 0, 0, 0,  # NW
                1, 1, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_L
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_R
            ]).reshape((6, 9)),

            # B0_GR_REP
            optionLbls[3]: np.array([
                0, 0, 0, 0, 0, 1, 0, 0, 0,  # NE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # NW
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_L
                1, 1, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_R
            ]).reshape((6, 9)),

            # B0_GL_ALT
            optionLbls[4]: np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # NE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                0, 0, 0, 0, 0, 1, 0, 0, 0,  # NW
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_L
                1, 1, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_R
            ]).reshape((6, 9)),

            # B0_GR_ALT
            optionLbls[5]: np.array([
                0, 0, 0, 0, 0, 1, 0, 0, 0,  # NE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # NW
                1, 1, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_L
                0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_R
            ]).reshape((6, 9)),
        }

        if "abstract" in agentClass:
            optionLbls += ["REP", "ALT"]

            s_init.update({
                optionLbls[6]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                optionLbls[7]: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            })

            s_term.update({
                optionLbls[6]: np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
                optionLbls[7]: np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
            })

            pi.update({
                # REP
                optionLbls[6]: np.array([
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # NE
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # NW
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_L
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_R
                    1, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_GL_REP
                    0, 1, 0, 0, 0, 0, 0, 0, 0,  # B0_GR_REP
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_GL_ALT
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_GR_ALT
                ]).reshape((10, 9)),

                # ALT
                optionLbls[7]: np.array([
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # NE
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # SW
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # NW
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_L
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_B1_R
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_GL_REP
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_GR_REP
                    0, 1, 0, 0, 0, 0, 0, 0, 0,  # B0_GL_ALT
                    1, 0, 0, 0, 0, 0, 0, 0, 0,  # B0_GR_ALT
                ]).reshape((10, 9))
            })

        if taskMode is "flat":
            for o in range(len(optionLbls)):
                s_init[o] = s_init[o][1:]
                s_term[o] = s_term[o][1:]
                pi[o] = pi[o][:, 1:]

    for o in optionLbls:
        params = {
            "s_init": pd.DataFrame(s_init[o], index=states).transpose(),
            "s_term": pd.DataFrame(s_term[o], index=states).transpose(),
            "pi": pd.DataFrame(
                pi[o], columns=states, index=allOptions[:pi[o].shape[0]]
            )
        }

        options[o] = Option(params)

    return options


def writeData(sim, dirName, fileName):
    filePath_csv = dirName + fileName + ".csv"
    filePath_Q = dirName + fileName + "_Q.npy"
    print("Writing data to {0}...".format(filePath_csv))
    with open(filePath_csv, 'a') as writeFile:
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

    np.save(filePath_Q, sim.data_Q)


def findState(state, env, value="index"):
    if value is "index":
        return env.states.index(state)
    elif value is "label":
        return env.state_lbls[env.states.index(state)]
