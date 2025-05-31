from .BaseAgent import BaseAgent
import chess
from collections import defaultdict

class ReBeLAgent(BaseAgent):

    def __init__(self, name):
        super().__init__(name)
        
        self.belief = []
        