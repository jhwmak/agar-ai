import pygame
import config as conf
from gamestate import GameState, start_game
from models.HeuristicModel import HeuristicModel
from models.RandomModel import RandomModel
from models.DQNModel import DQNModel

ai_models = [('Random', RandomModel(5, 10)), ('Heuristic', HeuristicModel())]
start_game(ai_models)