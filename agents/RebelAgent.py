from .BaseAgent import BaseAgent
import chess
from collections import defaultdict

class ReBeLAgent(BaseAgent):

    def __init__(self):
        super().__init__("ReBeLAgent")
        
        self.belief = []
        