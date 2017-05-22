# DRL
This is the repository for hosting for deep reinforcement learning project - CS533

I recommend using python 3.6 and suppose everyone uses Mac

Steps:
+ Try to install all the requirements in `setup.sh`
+ You can run the `main.py` file to see the examples of running files (contains the logic of the game). The result will display in Terminal. If you don't want to see the display of the game, just set the variable `is_render` in `main.py` to False. (for faster learning)
+ `agents` folder contains all the agents: `A3C.py`, `DDPG.py`, `DQN.py`, `ES.py` and `WIN.py` contain the template of the agent. You need to implement 3 functions:
  - `__init__`: initialize the class, can set up parameter for model
  - `select_action`: the environment will provide the observation, the agent should decide which action to do
  - `update`: update the perception of the agent by the observation, reward and done
  Note: I made an example in `DQN.py`, so you can follow that to make new Agent
+ To use the `visdom` for displaying the result:
  - You first need to start the server by running the command `python -m visdom.server`
  - Open the browser and navigate to the address: `http://localhost:8097`
+ `utils.py` contains all the supporting functions
+ `globals.py` import all import libraries, if you need something new, you can add to your own agent file.
+ `configs.py` contains all the configurations for different experiments
+ `Agents.py` and `Environments.py` contain the template for agents and environments respectively
