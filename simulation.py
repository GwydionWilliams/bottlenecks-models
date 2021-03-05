import numpy as np
from agent import Agent, Option
from environment import Environment
from sim_funs import buildEnv, defineOptions


class Simulation():
    def __init__(self, simParams, agentParams, envParams):
        self.taskMode = simParams["taskMode"]
        self.numTrials = simParams["numTrials"]
        self.regimes = envParams["regimes"]
        self.activeRegime = self.regimes[0]

        states = buildEnv(self.taskMode)
        envParams.update({"states": states})

        self.env = Environment(envParams)

        options = defineOptions(
            agentParams["class"],
            simParams["taskMode"],
            self.env.states.keys()
        )

        agentParams.update({"options": options})

        self.agent = Agent(agentParams, self.env)

        self.SG_sides = ["L", "R"]
        self.G_sides = ["L", "R"]

        self.data = {
            "alpha": [],
            "num_steps": [],
            "mu_steps": [],
            "sub_goal_side": [],
            "goal_side": [],
            "regime": []
        }

    def setupTrial(self):
        self.agent.sleep()
        if self.taskMode is "hierarchical":
            self.G_side = self.SG_sides[self.trialNum % 2]
            if self.activeRegime is "repeat":
                self.SG_side = self.G_side
            elif self.activeRegime is "alternate":
                self.SG_side = [
                    side for side in self.G_sides if side != self.G_side
                ][0]
        else:
            self.G_side = self.SG_sides[self.trialNum % 2]
            if self.G_side.contains("L"):
                self.SG_side = "L"
            else:
                self.SG_side = "R"

        self.env.placeReward(self.G_side, self.SG_side)

        self.t = 0

    def switchRegime(self):
        self.activeRegime = self.regimes[1]
        self.agent.step_counter = 0
        print("---------------------------- REGIME SWITCH --------------------"
              "--------")

    # def record_trial(self):
    #     self.data["alpha"].append(round(self.agent.alpha, 2))
    #     self.data["num_steps"].append(self.t)
    #     self.data["mu_steps"].append(np.round(np.mean(self.data["num_steps"])))
    #     self.data["sub_goal_side"].append(self.SG_side)
    #     self.data["goal_side"].append(self.G_side)
    #     self.data["regime"].append(self.activeRegime)
    #     self.data["action_history"].append("-".join(self.agent.action_history))
    #     self.data["state_history"].append("-".join(self.agent.state_history))
    #
    #     if self.agent.has_history:
    #         self.data_Q[:, :, :, self.trialNum] = self.agent.Q
    #     else:
    #         self.data_Q[:, :, self.trialNum] = self.agent.Q

    # def norm_Q(self):
    #     self.agent.Q[:, :] = self.agent.Q[:, :] / sum(sum(self.agent.Q[:, :]))

    # def summariseStep(self):
    #     print("t_{0}, S: {1}, S': {2}, A: {3}, "
    #           "r: {4}, SG_v: {5}".format(
    #               self.t,
    #               self.env.states[find_state(self.agent.prev_state, self.env)],
    #               self.env.states[find_state(self.agent.state, self.env)],
    #               self.agent.action_lbls[self.agent.a_i],
    #               self.agent.r,
    #               self.agent.SG_visited)
    #           )
    #     # print(np.round(self.agent.Q, 3))

    # def summarise_trial(self):
    #     if self.taskMode is "hierarchical":
    #         print("steps taken this trial: {0}, mean steps taken = {1},\n"
    #               "Q(B0L, :) = {2}, \nQ(B0R, :) = {3}, \nQ(B1, :) = {4}".
    #               format(
    #                   self.t,
    #                   self.mu_steps[-1],
    #                   np.round(self.agent.Q[:, 0], 3),
    #                   np.round(self.agent.Q[:, 1], 3),
    #                   np.round(self.agent.Q[:, 5], 3),
    #               ))
    #
    #     elif self.taskMode is "flat":
    #         print("steps taken this trial: {0}, mean steps taken = {1},\n"
    #               "Q(B0, :) = {2}, \nQ(B1, :) = {3}".
    #               format(
    #                   self.t,
    #                   self.mu_steps[-1],
    #                   np.round(self.agent.Q[:, 0], 3),
    #                   np.round(self.agent.Q[:, 4], 3),
    #               ))
    #
    # def summarise_chunk(self):
    #     if self.taskMode is "flat":
    #         if self.agent.has_history:
    #             print("-------------------------------------------------------"
    #                   "----------------\n"
    #                   "        trial num = {0}\n"
    #                   " mean steps taken = {1}\n"
    #                   "     Q(B0, B0, :) = {2}\n"
    #                   "    Q(B0, SGL, :) = {3}\n"
    #                   "    Q(B0, SGR, :) = {4}\n"
    #                   "    Q(SGL, B1, :) = {5}\n"
    #                   "    Q(SGR, B1, :) = {6}\n"
    #                   "-------------------------------------------------------"
    #                   "-----------------".format(
    #                       self.trialNum, self.data["mu_steps"][-1],
    #                       np.round(self.agent.Q[0, :, 0], 2),
    #                       np.round(self.agent.Q[0, :, 1], 2),
    #                       np.round(self.agent.Q[0, :, 2], 2),
    #                       np.round(self.agent.Q[1, :, 4], 2),
    #                       np.round(self.agent.Q[2, :, 4], 2))
    #                   )
    #         else:
    #             print("-------------------------------------------------------"
    #                   "----------------\n"
    #                   "        trial num = {0}\n"
    #                   " mean steps taken = {1}\n"
    #                   "         Q(B0, :) = {2}\n"
    #                   "        Q(SGL, :) = {3}\n"
    #                   "        Q(SGR, :) = {4}\n"
    #                   "         Q(B1, :) = {5}\n"
    #                   "-------------------------------------------------------"
    #                   "-----------------".format(
    #                       self.trialNum, self.data["mu_steps"][-1],
    #                       np.round(self.agent.Q[:, 0], 2),
    #                       np.round(self.agent.Q[:, 1], 2),
    #                       np.round(self.agent.Q[:, 2], 2),
    #                       np.round(self.agent.Q[:, 4], 2))
    #                   )
    #     elif self.taskMode is "hierarchical":
    #         if self.agent.has_history:
    #             print("------------------------------------------------------"
    #                   "-----------------\n"
    #                   "        trial num = {0}\n"
    #                   " mean steps taken = {1}\n"
    #                   "     Q(0, B0L, :) = {2}\n"
    #                   "     Q(0, B0R, :) = {3}\n"
    #                   "   Q(B0L, SGL, :) = {4}\n"
    #                   "   Q(B0R, SGL, :) = {5}\n"
    #                   "   Q(B0L, SGR, :) = {6}\n"
    #                   "   Q(B0R, SGR, :) = {7}\n"
    #                   "    Q(SGL, B1, :) = {8}\n"
    #                   "    Q(SGR, B1, :) = {9}\n"
    #                   "-------------------------------------------------------"
    #                   "----------------".format(
    #                       self.trialNum, self.data["mu_steps"][-1],
    #                       np.round(self.agent.Q[0, :, 0], 2),
    #                       np.round(self.agent.Q[1, :, 1], 2),
    #                       np.round(self.agent.Q[0, :, 2], 2),
    #                       np.round(self.agent.Q[1, :, 2], 2),
    #                       np.round(self.agent.Q[0, :, 3], 2),
    #                       np.round(self.agent.Q[1, :, 3], 2),
    #                       np.round(self.agent.Q[2, :, 5], 2),
    #                       np.round(self.agent.Q[3, :, 5], 2))
    #                   )
    #             # print(np.round(self.agent.Q[:, :, :], 4))
    #         else:
    #             print("-------------------------------------------------------"
    #                   "----------------\n"
    #                   "        trial num = {0}\n"
    #                   " mean steps taken = {1}\n"
    #                   "        Q(B0L, :) = {2}\n"
    #                   "        Q(B0R, :) = {3}\n"
    #                   "        Q(SGL, :) = {4}\n"
    #                   "        Q(SGR, :) = {5}\n"
    #                   "         Q(B1, :) = {6}\n"
    #                   "-------------------------------------------------------"
    #                   "----------------".format(
    #                       self.trialNum, self.data["mu_steps"][-1],
    #                       np.round(self.agent.Q[:, 0], 2),
    #                       np.round(self.agent.Q[:, 1], 2),
    #                       np.round(self.agent.Q[:, 2], 2),
    #                       np.round(self.agent.Q[:, 3], 2),
    #                       np.round(self.agent.Q[:, 5], 2))
    #                   )
