import Arena
from MCTS import MCTS
from connect4.Connect4Players import *
from Coach import Coach
from connect4.Connect4Game import Connect4Game
from connect4.tensorflow.NNet import NNetWrapper as NNet
from utils import dotdict

import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
 # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

g = Connect4Game()

# all players
rp = RandomPlayer(g).play
gp = OneStepLookaheadConnect4Player(g).play
hp = HumanConnect4Player(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n1.load_checkpoint('./pretrained_models/connect4/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=Connect4Game.display)

print(arena.playGames(2, verbose=True))
