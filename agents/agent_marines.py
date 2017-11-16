
from pysc2.agents import base_agent
from pysc2.lib import actions


class MarineFocusedAgent(base_agent.BaseAgent):
    """ Agent focuses on assisting Marines in win. """

    def step(self, obs):
        print(obs.observation["available_actions"])
        return actions.FunctionCall(obs.observation["available_actions"][0], [])