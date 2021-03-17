import csv
import numpy as np
import pandas as pd
import copy
from agent import Option, PrimitiveAction


def buildEnv():
    states = {
        "B0L":  [0, 0, 0],
        "B0R":  [0, 0, 1],
        "SGL": [-1, 1, 0],
        "SGR":  [1, 1, 0],
        "DL1": [-2, 2, 0],
        "B1":   [0, 2, 0],
        "DR1":  [2, 2, 0],
        "VL":  [-1, 3, 0],
        "VR":   [1, 3, 0],
        "DL2": [-2, 4, 0],
        "GL":   [0, 4, 1],
        "GR":   [0, 4, 0],
        "DR2":  [2, 4, 0]
    }

    return states


def definePrimitiveActions(states):
    actionLbls = ["NE", "SE", "SW", "NW"]

    s_init = {  # B0L B0R SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
        actionLbls[0]: [  # NE
            0, 0, 0, 0, None, 0, None, 0, 0,
            None, None, None, None
        ],
        actionLbls[1]: [  # SE
            None, None, 0, None, 0, 0,
            None, 0, None, 0, None, 0, None
        ],
        actionLbls[2]: [  # SW
            None, None, None, 0, None,
            0, 0, None, 0, None, 0, None, 0
        ],
        actionLbls[3]: [  # NW
            0, 0, 0, 0, None, 0, None, 0, 0,
            None, None, None, None
        ]
    }

    s_term = {  # B0L B0R SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
        actionLbls[0]: [None, None, None, 1, None, 1, 1, None, 1, None, 1, None, 1],
        actionLbls[1]: [1, 1, 1, 1, None, 1, None, 1, 1, None, None, None, None],
        actionLbls[2]: [1, 1, 1, 1, None, 1, None, 1, 1, None, None, None, None],
        actionLbls[3]: [None, None, 1, None, 1, 1, None, 1, None, 1, None, 1, None]
    }

    level = {
        actionLbls[0]: 0,
        actionLbls[1]: 0,
        actionLbls[2]: 0,
        actionLbls[3]: 0
    }

    numActions = len(actionLbls)
    numStates = len(s_init[actionLbls[0]])

    pi = {}
    for i, a in enumerate(actionLbls):
        pi[a] = np.empty((numActions, numStates))
        pi[a][:] = None

        for s in range(numStates):
            if s_init[a][s] == 1:
                pi[a][i, s] = 1

    primitiveActions = {}
    for a in actionLbls:
        params = {
            "s_init": pd.DataFrame(s_init[a], index=states).transpose(),
            "s_term": pd.DataFrame(s_term[a], index=states).transpose(),
            "pi": pd.DataFrame(pi[a], columns=states, index=actionLbls),
            "level": level[a]
        }

        primitiveActions[a] = PrimitiveAction(params)

    return primitiveActions


def defineOptions(agentClass, states):
    options = definePrimitiveActions(states)

    if "hierarchical" in agentClass:
        optionLbls = ["B_B_L", "B_B_R", "B_B_L_REV", "B_B_R_REV",
                      "B0_GL_REP", "B0_GR_REP",
                      "B0_GL_ALT", "B0_GR_ALT"]

        allOptions = list(options.keys()) + optionLbls

        univInit = np.array(
            [0, 0, None, None, None, None, None,
                None, None, None, None, None, None]
        )

        s_init = {  # B0L B0R SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
            optionLbls[0]: np.array(
                [0, 0, None, None, None, 0, None,
                 None, None, None, None, None, None]
            ),
            optionLbls[1]: np.array(
                [0, 0, None, None, None, 0, None,
                 None, None, None, None, None, None]
            ),
            optionLbls[2]: np.array(
                [None, None, None, None, None,
                 0, None, None, None, None, 0, None, None]
            ),
            optionLbls[3]: np.array(
                [None, None, None, None, None,
                 0, None, None, None, None, None, 0, None]
            ),
            optionLbls[4]: copy.deepcopy(univInit),
            optionLbls[5]: copy.deepcopy(univInit),
            optionLbls[6]: copy.deepcopy(univInit),
            optionLbls[7]: copy.deepcopy(univInit)
        }

        s_term = {  # BNoneL BNoneR SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
            optionLbls[0]: np.array([None, None, None, None, None, 1, None, None, None, None, 1, None, None]),
            optionLbls[1]: np.array([None, None, None, None, None, 1, None, None, None, None, None, 1, None]),
            optionLbls[2]: np.array([1, 1, None, None, None, 1, None, None, None, None, None, None, None]),
            optionLbls[3]: np.array([1, 1, None, None, None, 1, None, None, None, None, None, None, None]),
            optionLbls[4]: np.array([None, None, None, None, None, None, None, None, None, None, 1, None, None]),
            optionLbls[5]: np.array([None, None, None, None, None, None, None, None, None, None, None, 1, None]),
            optionLbls[6]: np.array([None, None, None, None, None, None, None, None, None, None, 1, None, None]),
            optionLbls[7]: np.array([None, None, None, None, None, None,
                                     None, None, None, None, None, 1, None])
        }

        level = {
            optionLbls[0]: 1,
            optionLbls[1]: 1,
            optionLbls[2]: 1,
            optionLbls[3]: 1,
            optionLbls[4]: 2,
            optionLbls[5]: 2,
            optionLbls[6]: 2,
            optionLbls[7]: 2
        }

        pi = {  # B0L B0R SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
            # B_B_L
            optionLbls[0]: np.array([
                None, None, 1,    None, None, None, None, 1,    None, None, None, None, None,  # NE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                1,    1,    None, None, None, 1,    None, None, None, None, None, None, None,  # NW
            ]).reshape((4, 13)),

            # B_B_R
            optionLbls[1]: np.array([
                1,    1,    None, None, None, 1,    None, None, None, None, None, None, None,  # NE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                None, None, None, 1,    None, None, None, None, 1,    None, None, None, None,  # NW
            ]).reshape((4, 13)),

            # B_B_L_REV
            optionLbls[2]: np.array([
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                None, None, 1,    None, None, None, None, 1,    None, None, None, None, None,  # SE
                None, None, None, None, None, 1,    None, None, None, None, 1,    None, None,  # SW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
            ]).reshape((4, 13)),

            # B_B_R_REV
            optionLbls[3]: np.array([
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                None, None, None, None, None, 1,    None, None, None, None, None, 1,    None,  # SE
                None, None, None, 1,    None, None, None, None, 1,    None, None, None, None,  # SW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
            ]).reshape((4, 13)),

            # B0_GL_REP
            optionLbls[4]: np.array([
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
                1,    1,    None, None, None, 1,    None, None, None, None, None, None, None,  # B_B_L
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L_REV
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R_REV
            ]).reshape((8, 13)),

            # B0_GR_REP
            optionLbls[5]: np.array([
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L
                1,    1,    None, None, None, 1,    None, None, None, None, None, None, None,  # B_B_R
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L_REV
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R_REV
            ]).reshape((8, 13)),

            # B0_GL_ALT
            optionLbls[6]: np.array([
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
                None, None, None, None, None, 1,    None, None, None, None, None, None, None,  # B_B_L
                1,    1,    None, None, None, None, None, None, None, None, None, None, None,  # B_B_R
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L_REV
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R_REV
            ]).reshape((8, 13)),

            # B0_GR_ALT
            optionLbls[7]: np.array([
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
                1,    1,    None, None, None, None, None, None, None, None, None, None, None,  # B_B_L
                None, None, None, None, None, 1,    None, None, None, None, None, None, None,  # B_B_R
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L_REV
                None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R_REV
            ]).reshape((8, 13)),
        }

        if "abstract" in agentClass:
            optionLbls += ["REP", "ALT"]

            s_init.update({  # B0L B0R SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
                optionLbls[-2]: copy.deepcopy(univInit),
                optionLbls[-1]: copy.deepcopy(univInit)
            })

            s_term.update({  # B0L B0R SGL SGR DL1 B1 DR1 VL VR DL2 GL GR DR2
                optionLbls[-2]: np.array(
                    [None, None, None, None, None, None,
                        None, None, None, None, 1, 1, None]
                ),
                optionLbls[-1]: np.array(
                    [None, None, None, None, None, None,
                        None, None, None, None, 1, 1, None]
                ),
            })

            level.update({
                optionLbls[-2]: 3,
                optionLbls[-1]: 3
            })

            pi.update({
                # REP
                optionLbls[-2]: np.array([
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_B1_L
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_B1_R
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L_REV
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R_REV
                    1,    None, None, None, None, None, None, None, None, None, None, None, None,  # B0_GL_REP
                    None, 1,    None, None, None, None, None, None, None, None, None, None, None,  # B0_GR_REP
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_GL_ALT
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_GR_ALT
                ]).reshape((12, 13)),

                # ALT
                optionLbls[-1]: np.array([
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # NE
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # SE
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # SW
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # NW
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_B1_L
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_B1_R
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_L_REV
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B_B_R_REV
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_GL_REP
                    None, None, None, None, None, None, None, None, None, None, None, None, None,  # B0_GR_REP
                    None, 1,    None, None, None, None, None, None, None, None, None, None, None,  # B0_GL_ALT
                    1,    None, None, None, None, None, None, None, None, None, None, None, None,  # B0_GR_ALT
                ]).reshape((12, 13))
            })

        for o in optionLbls:
            params = {
                "s_init": pd.DataFrame(s_init[o], index=states).transpose(),
                "s_term": pd.DataFrame(s_term[o], index=states).transpose(),
                "pi": pd.DataFrame(
                    pi[o], columns=states, index=allOptions[:pi[o].shape[0]]
                ),
                "level": level[o]
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
