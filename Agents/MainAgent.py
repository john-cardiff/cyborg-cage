from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from Agents.DQNAgent import DQNAgent
from Agents.BlueSleepAgent import BlueSleepAgent


class MainAgent(BaseAgent):
    def __init__(self):
        self.end_episode()
        # wake_up() and wake_up_value is to handle a rare case
        # it remembers the previous agent it assigned in case the red agent goes back to sleep thinking it is facing a Sleep agent
        self.wake_up_value = 0
        self.previous_agent = "Sleep"

    def get_action(self, observation, action_space=None):
        self.wake_up_value = self.wake_up_value - 1
        previous_two_observations = list(observation + self.last_observation)
        # pick agent based on a fingerprint
        self.assign_agent(previous_two_observations)
        self.last_observation = observation
        if self.agent_name == "Sleep" and self.wake_up_value > 97 and sum(observation) > 0:
            self.wake_up()
        return self.agent.get_action(observation)

    def assign_agent(self, previous_two_observations):

        # fingerprints are sum of both previous observation bits
        sleep_fingerprinted = [0] * 52
        meander_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if previous_two_observations == meander_fingerprinted:
            self.use_meander()

        elif previous_two_observations == bline_fingerprinted or previous_two_observations == bline_fingerprinted_2:
            self.use_bline()

        # stick with sleep
        elif previous_two_observations == sleep_fingerprinted:
            # entering a sleep through fingerprint remembers the previous agent it assigned
            self.wake_up_value = 100
            self.previous_agent = self.agent_name
            self.end_episode()
        else:
            pass

    def wake_up(self):
        if self.previous_agent == "Meander":
            self.use_meander()
            self.wake_up_value = 0
        elif self.previous_agent == "Bline":
            self.use_bline()
            self.wake_up_value = 0
        else:
            pass

    def use_meander(self):
        self.agent = DQNAgent(chkpt_dir="../Models/model_meander/", algo='DDQNAgent', env_name='Scenario1b')
        # needed to get the pytorch checkpoint
        self.agent.load_models()
        self.agent_name = "Meander"

    def use_bline(self):
        self.agent = DQNAgent(chkpt_dir="../Models/model_b_line/", algo='DDQNAgent', env_name='Scenario1b')
        # needed to get the pytorch checkpoint
        self.agent.load_models()
        self.agent_name = "Bline"

    def train(self):
        pass

    def end_episode(self):
        self.last_observation = [0] * 52
        # we start with the sleep agent, we might want to build another one though as the default
        self.agent = BlueSleepAgent()
        self.agent_name = "Sleep"

    def set_initial_values(self, action_space, observation):
        pass
