##
##                  Additional Pylons
##
##          Beau, Max, Roger, Nathan, David
##              Software Engineering II
##                    Dr. Mengel
##                     Fall 2017
##
##  Agent file:
##
##  This python file contains the code for our team's learning agent
##  created to play the game StarCraft II. This agent uses a QLearning
##  table to learn how to fight against varied enemies.
##
##
## Follows tutorial from https://chatbotslife.com/building-a-smart-pysc2-agent-cdc269cb095d
## This agent is created specially for combat situations for the Terran race.

import math
import numpy as np
import pandas as pd
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# These variables store how the agent can send actions to the game
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_HOSTILE = 4

# The names of the possible actions
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_ATTACK_RANDOM_ENEMY = 'attack_randomenemy'
ACTION_ATTACK_TOP_LEFT = 'attack_topleft'
ACTION_ATTACK_TOP_RIGHT = 'attack_topright'
ACTION_ATTACK_BOTTOM_LEFT = 'attack_bottomleft'
ACTION_ATTACK_BOTTOM_RIGHT = 'attack_bottomright'

# smart_actions is used as the blueprint to the QLearning table
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK_RANDOM_ENEMY,
    ACTION_ATTACK_TOP_LEFT,
    ACTION_ATTACK_TOP_RIGHT,
    ACTION_ATTACK_BOTTOM_LEFT,
    ACTION_ATTACK_BOTTOM_RIGHT
]


# Code for QLearning table from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions      # the list of possible actions, made from smart_actions
        self.lr = learning_rate     # indicates the importance of the new information
        self.gamma = reward_decay   # indicates how much the older info will influence new information
        self.epsilon = e_greedy     # the chance that agent will not pick a random action
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        # If the state is not already in the table, create a new one
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:  # If the random number is less than the chance of a random action
            # choose best action from the QLearn table
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value, so choose a random actoin
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax(axis=1) # get the name of the best action
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update the table with the learned information
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append the new state row to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        # Killed unit score and army supply to be used in reward calculation
        self.previous_killed_unit_score = 0
        self.previous_army_supply = 0

        self.previous_action = None
        self.previous_state = None

        # From Base Agent
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None


    def step(self, obs):
        super(SmartAgent, self).step(obs)

        army_supply = obs.observation['player'][5]
        killed_unit_score = obs.observation['score_cumulative'][5]

        # Enemy corners keeps track of which corners of the sceen enemies are currently present
        enemy_corners = [0,0,0,0]
        # This command grabs the coordinates of all enemies currently visible
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        enemy_positions = [(enemy_x[i], enemy_y[i]) for i in range(len(enemy_y))]

        for i in range(0, len(enemy_y)):

            y = int(math.ceil((enemy_y[i]) // 36))
            x = int(math.ceil((enemy_x[i]) // 36))
            enemy_corners[x + (y * 2)] = 1  # indicates which quarter of the screen an enemy is in,
                                          # [top left, top right, bottom left, bottom right]
        enemy_count = len(enemy_y)

        # The current state is used as a row to the QLearning table
        current_state = [
            army_supply,
            enemy_count,
            enemy_corners[0],
            enemy_corners[1],
            enemy_corners[2],
            enemy_corners[3]

        ]

        # Reward calculation is based on army supply, enemies, and the enemy-corners
        reward = 0
        if self.previous_action is not None:

            if killed_unit_score > self.previous_killed_unit_score:
                # Increase the reward by the same amount as the killed unit score
                reward += killed_unit_score - self.previous_killed_unit_score
            elif army_supply < self.previous_army_supply:
                # Decrease the reward proportional to the value of the unit that died
                reward -= (self.previous_army_supply - army_supply) * 50

            # Add the new information to the QLearning table
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        # Choose an action based on the Q table and get its action name
        chosen_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[chosen_action]

        # Update the information about the previous state and action
        self.previous_killed_unit_score = killed_unit_score
        self.previous_army_supply = army_supply
        self.previous_state = current_state
        self.previous_action = chosen_action

        # This if-else block tells the game how to act out the chosen action
        if '_' in smart_action:     # If the chosen action had an argument after it
            smart_action, arg = smart_action.split('_')

        if smart_action == ACTION_DO_NOTHING:   # If the agent chose to do nothing
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_ARMY:    # If the agent chose to select its army
            if _SELECT_ARMY in obs.observation['available_actions']:    # Check to see if that action is available
                return actions.FunctionCall(_SELECT_ARMY, [[0]])

        elif smart_action == ACTION_ATTACK:     # If the agent chose to attack
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:  # Check that attacking is available
                attack_pos = (0, 0)
                # Give the attack the appropriate parameters based on the argument to the attack action
                if arg == 'topleft':
                    attack_pos = (25, 32)
                elif arg == 'topright':
                    attack_pos = (39, 32)
                elif arg == 'bottomleft':
                    attack_pos = (25, 40)
                elif arg == 'bottomright':
                    attack_pos = (39, 40)
                elif arg == 'randomenemy':  # If the agent chose to select an enemy to attack
                    if len(enemy_positions) > 0:    # Check that there are enemies on the map
                        attack_pos = enemy_positions[0]     # Get the coordinates of a random enemy

                return actions.FunctionCall(_ATTACK_MINIMAP, [[0], (attack_pos[0], attack_pos[1])])

        return actions.FunctionCall(_NO_OP, [])