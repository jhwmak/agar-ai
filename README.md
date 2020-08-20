# Agar.ai

Final project for CIS 522: Deep Learning at the University of Pennsylvania.

Python-based implementation of agar.io in Pygame and a set of models built in vanilla Python and PyTorch for learning to play this game with the objective of maximizing change in mass.

---

### Introduction

Agar.io is an online multiplayer game in which players control an agent with the objective ofmaximizing their score by consuming various objects including food, viruses, mass pellets, and other players. We built a variety of reinforcement learning models and other agents on a custom Python-based implementation of the Agar.io game.

We compared the models to better understand the impact that encoding biases into policies and architectures has on performance, both for the "full" game (a complete environment with other agents, viruses, and food) and subsets of the game(e.g., an environment with only food and no other agents). Our implementations range in their level of bias, from a strongly biased heuristic model with a hard-coded greedy policy, to a semi-biased deep reinforcement learning model which uses input vectors representing encoded game state, to a relatively unbiased deep reinforcement learning model which takes in screenshots of the GUI andencodes game state using a CNN.

---

### Repo structure

```text
.
├── demo/
|   └── *                        GIFs and Videos demoing agents
|
├── model_utils/                 Code shared by models
|   ├── train_utils.py           Shared functions by the two training scripts
|   ├── fs_utils.py              Helper functions for saving
|   └── ReplayBuffer.py          Replay buffer datastructure
|
├── models/                      Agar.io models
|   ├── DeepCNNModel.py          CNN-based RL agent on screenshot state
|   ├── DeepRLModel.py           FC-based RL agent on encoded state
|   ├── HeuristicModel.py        Greedy baseline model
|   ├── ModelInterface.py        Shared interface for all models
|   └── RandomModel.py           Random action baseline model
|
├── plots/
|   ├── important/               Best performing plots
|   └── *                        Plots generated over the course of training and test
|
├── store/
|   └── nets/                    Nets generated over the course of training
|       ├── food_trained_cnn.pt                        CNN-based RL agent for only food
|       ├── food_trained_drl.pt                        FC-based RL (DRL) agent for only food
|       ├── full_food_pretrained_trained_cnn.pt        CNN-based RL agent for enemies (pretrained from food_trained_cnn)
|       └── full_from_scratch_trained_cnn.pt           CNN-based RL agent for enemies (trained from scratch)
|
├── __test-encoded-state__.py    Tests that RL agent state encoding works as expected
├── __test-fsutils__.py          Tests that saving and loading net params works as expected
|
├── actions.py                   Set of allowed actions for an agent to take
├── agario.py                    Script for running game manually in GUI
├── agent.py                     Agent and AgentCell classes
├── camera.py                    Handles moving viewport in GUI to follow agent
├── config.py                    Set of constants used throughout the repo
├── evaluate.py                  Plots performance of set of models
├── food.py                      Food object
├── gamestate.py                 Implementation of headless game state
├── mass.py                      Mass pellet object
├── test.py                      Run trained model on GUI
├── train_cnn.py                 Train CNN model
├── train_drl.py                 Train RL model
├── utils.py                     Helper functions for dealing with object interactions
├── virus.py                     Virus object
|
├── requirements.txt             Python dependencies
├── .gitignore                   Files to not include in git
└── README.md                    Documentation
```

---

### Running the code

First, clone this repo and change directories into the root of the repo.

#### Installing dependencies

You need python3 (and pip3) installed to run our project.

To install dependencies, run:

```bash
pip3 install -r requirements.txt
```

If you have errors along the lines of `Could not build wheels for ..., since package 'wheel' is not installed.`, run `pip3 install wheel` and then try re-running the install command.

Pygame does not always install correctly on MacOS Mojave and more recent. Even if pip says that the dependency was added, when it comes to actually running a file which imports Pygame it may not actually perform as expected (i.e. you won't be able to see the GUI for the game). To get around this, you can follow [this SO post](https://stackoverflow.com/questions/52718921/problems-getting-pygame-to-show-anything-but-a-blank-screen-on-macos-mojave) to install Pygame from source.

#### Running locally

To play the game, run:

```bash
python3 agario.py
```

To modify the enemies, modify the `ai_models` variable in `agario.py`.

<br />

To train the DRL model, run:

```bash
python3 train_drl.py
```

To train the CNN-based model, run:

```bash
python3 train_cnn.py
```

**Training and model hyperparameters can be modified at the top of each train file.**

<br />

To run the game with a specific model as the main player, first select a model you wish to run. The name description of each model can be found in the file tree above in the `store/nets` directory. Once selected, run:

```bash
python3 test.py [drl | cnn] [model_name]
```

For example, to run `food_trained_cnn.pt`, run:

```bash
python3 test.py cnn food_trained_cnn
```

#### Running on AWS SageMaker

To run on AWS Sagemaker, create a notebook instance in the SageMaker console and associate this repository with the notebook. Documentation on how to do this can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-repo.html). Once your instance has been set up, click open Jupyter.

Once open, create a new notebook and be sure to install dependencies. Finally, simply import the train file for the model you wish to train.

Note that GUIs will not appear in SageMaker.

---

### Demo

DRL agent eating food:

![DRL](demo/drl-food.gif)

CNN agent eating food:

![CNN](demo/cnn-food.gif)

---

