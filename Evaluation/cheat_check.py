import inspect
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

# ADDED AGENT HERE
from Agents.MainAgent import MainAgent

# ADDED FOR REPRODUCING RESULTS
import random
random.seed(1)

num_steps = 30
MAX_EPS = 10
agent_name = 'Blue'

# creates a random order for the agents to appear in
red_agents_order = MAX_EPS * [SleepAgent] + MAX_EPS * [B_lineAgent] + MAX_EPS * [RedMeanderAgent]
random.shuffle(red_agents_order)
red_agents_score = {"Sleep": 0, "B_line": 0, "Meander": 0}

# CHANGED WRAPPER HERE
def wrap(env):
    return ChallengeWrapper(agent_name, env)

if __name__ == "__main__":

    agent = MainAgent()

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    for red_agent in red_agents_order:
        cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
        wrapped_cyborg = wrap(cyborg)
        observation = wrapped_cyborg.reset()
        action_space = wrapped_cyborg.get_action_space(agent_name)

        r = 0
        for j in range(num_steps):
            action = agent.get_action(observation, action_space)
            observation, rew, done, info = wrapped_cyborg.step(action)
            r += rew

        if red_agent == SleepAgent:
            red_agents_score["Sleep"] += r
        elif red_agent ==  RedMeanderAgent:
            red_agents_score["Meander"] += r
        else:
            red_agents_score["B_line"] += r
        observation = wrapped_cyborg.reset()

    # it would be better if we calculated the mean and the scores were a list (or numpy array)
    # but this is sufficient given the purpose
    print(red_agents_score)