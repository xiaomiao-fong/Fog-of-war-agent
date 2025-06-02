import pygame as p
import sys
from multiprocessing import Process, Queue
import chess
from chess import Move
import random
from utils.BoardManipulation import *
import agents
import time

def test_win_rate(agent, num_games=100):