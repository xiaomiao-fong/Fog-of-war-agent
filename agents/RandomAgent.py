import chess
from chess import Move
from .BaseAgent import BaseAgent
import time
import random

class RandomAgent(BaseAgent):

    def __init__(self, name):
        super().__init__(name)

    def act(self, board : chess.Board) -> Move:
        return random.choice(list(board.pseudo_legal_moves.__iter__()))