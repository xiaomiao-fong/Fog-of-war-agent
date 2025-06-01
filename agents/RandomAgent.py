import chess
from chess import Move
from .BaseAgent import BaseAgent
import random

class RandomAgent(BaseAgent):

    def __init__(self):
        super().__init__("RandomAgent")

    def act(self, board : chess.Board) -> Move:
        return random.choice(list(board.pseudo_legal_moves))