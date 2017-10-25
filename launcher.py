# This utility helps launch a launcher.options file.
# To use:
# 1. Create a file called "launcher.options" right next to this file.
# 2. On the first line of the file, add the name of the map to launch.
#    The map should be found in the "maps" directory. Example: "CollectMineralShards"
# 3. On the second line of the file, add the name of the agent to run.
#    The agent should be located in the "agents" directory. Example: "Teran.scripted_agent.CollectMineralShards"
# 4. On the third line of the file, place any additional arguments.
#    Example: "--agent_race T --screen_resolution 84"
# 4. Run launcher.py. Also Python should be in the path.

import sys
import os

from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.maps import lib

import gflags
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

class GithubGame(lib.Map):
    directory = (os.path.dirname(os.path.realpath(__file__)) if os.path.dirname(os.path.realpath(__file__)).
                 endswith("\\") else os.path.dirname(os.path.realpath(__file__)) + "\\") + "maps\\"
    download = "https://github.com/AdditionalPylons2017/StarcraftAI"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


def run_thread(agent_cls, map_name, args):
    with sc2_env.SC2Env(
      map_name=map_name,
      agent_race=args["agent_race"],
      bot_race=args["bot_race"],
      difficulty=args["difficulty"],
      step_mul=args["step_mul"],
      game_steps_per_episode=args["game_steps_per_episode"],
      screen_size_px=(args["screen_resolution"], args["screen_resolution"]),
      minimap_size_px=(args["minimap_resolution"], args["minimap_resolution"]),
      visualize=True) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = agent_cls()
        run_loop.run_loop([agent], env, args["max_agent_steps"])
        if args["save_replay"]:
            env.save_replay(agent_cls.__name__)


def getopts(argv2):
    opts = {"render": True, "screen_resolution": 84, "minimap_resolution": 64, "max_agent_steps": 2500, "game_steps_per_episode": 0, "step_mul": 8,
            "agent": "pysc2.agents.random_agent.RandomAgent", "agent_race": None, "bot_race": None, "difficulty": None, "profile": False, "trace": False,
            "parallel": 1, "save_replay": True}
    if argv2 is None or len(argv2) == 0:
        return opts
    argv = argv2.split(" ")
    optsTypes = {"render": "bool", "screen_resolution": "int", "minimap_resolution": "int", "max_agent_steps": "int",
            "game_steps_per_episode": "int", "step_mul": "int",
            "agent": "string", "agent_race": "race", "bot_race": "race", "difficulty": "difficulty",
            "profile": "bool", "trace": "bool",
            "parallel": "int", "save_replay": "bool"}
    while argv:
        if argv[0][0] == '-':
            key = argv[0]
            if key.startswith('--'):
                key = key[2:]
            if optsTypes[key] == "bool":
                opts[key] = True if argv[1] == "1" else False
            elif optsTypes[key] == "int":
                opts[key] = int(argv[1])
            elif optsTypes[key] == "race":
                opts[key] = argv[1]
            elif optsTypes[key] == "difficulty":
                opts[key] = argv[1]
            else:
                opts[key] = argv[1]
        argv = argv[1:]
    return opts


def find_starcraft_minigame_dir():
    if os.path.isdir("D:\Program Files (x86)\StarCraft II\Maps\mini_games\\"):
        return "D:\Program Files (x86)\StarCraft II\Maps\mini_games\\"
    elif os.path.isdir("C:\Program Files (x86)\StarCraft II\Maps\mini_games\\"):
        return "C:\Program Files (x86)\StarCraft II\Maps\mini_games\\"
    elif os.path.isdir("C:\Program Files\StarCraft II\Maps\mini_games\\"):
        return "C:\Program Files\StarCraft II\Maps\mini_games\\"
    elif os.path.isdir("D:\Program Files\StarCraft II\Maps\mini_games\\"):
        return "D:\Program Files\StarCraft II\Maps\mini_games\\"
    else:
        raise EnvironmentError("Cannot find StarCraft mini game directory.")


def main():
    mapFileFull = mapFile = agentFile = args = ""
    mamaPath = os.path.dirname(os.path.realpath(__file__))

    if not mamaPath.endswith("\\"):
        mamaPath = mamaPath + "\\"

    starcraft_dir = find_starcraft_minigame_dir()

    if not os.path.exists("launcher.options"):
        raise EnvironmentError("Cannot find launcher.options. See header of launcher.py on how to use.");

    with open("launcher.options") as f:
        mapFile = f.readline().replace("\n", "")
        agentFile = f.readline().replace("\n", "")
        args = f.readline().replace("\n", "")

    if not mapFile.endswith(".SC2Map"):
        mapFileFull = mapFile + ".SC2Map"

    if os.path.exists(mapFileFull):
        globals()[mapFile] = type(mapFile, (GithubGame,), dict(filename=mapFile))

    sys.path.append(mamaPath + "agents\\" + agentFile.split(".")[0])
    import importlib
    from pysc2 import maps
    module_name, classname = ("agents." + agentFile).rsplit(".", 1)
    mod = getattr(importlib.import_module(module_name), classname)
    maps.get(mapFile)
    run_thread(mod, mapFile, getopts(args))

    os.system('pause')



if  __name__ =='__main__':main()