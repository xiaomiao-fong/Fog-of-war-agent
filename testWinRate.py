import pygame as p
import sys
from multiprocessing import Process, Queue
import chess
from chess import Move
import random
from utils.BoardManipulation import *
import agents
import time

def test_win_rate(agent1, agent2, num_games=100):
    white_wins = 0
    black_wins = 0
    draws = 0

    for _ in range(num_games):
        board = chess.Board()
        agents = [agent1, agent2]
        current_agent_index = 0

        while not board.is_game_over():
            current_agent = agents[current_agent_index]
            move = current_agent.act(board)
            board.push(move)
            current_agent_index = 1 - current_agent_index  # Switch to the other agent

        if board.king(chess.WHITE) is None:
            result = '0-1'
        elif board.king(chess.BLACK) is None:
            result = '1-0'
        else:
            result = '1/2-1/2'

        if result == '1-0':
            white_wins += 1
        elif result == '0-1':
            black_wins += 1
        else:
            draws += 1
        # check if King is not in board

        
        #save move history to file
        # with open("game_history.txt", "a") as f:
        #     f.write(f"Game {_ + 1}:\n")
        #     f.write(str(board) + "\n")
        #     f.write(f"Result: {result}\n\n")



    print(f"Results after {num_games} games:")
    print(f"White:{agent1.name}, wins: {white_wins}")
    print(f"Black:{agent2.name}, wins: {black_wins}")
    return white_wins, black_wins, draws

if __name__ == "__main__":
    # Initialize agents
    agent1 = agents.GreedyAgent(faction=chess.WHITE)
    agent2 = agents.RandomAgent()

    # Test win rate
    num_games = 100
    test_win_rate(agent1, agent2, num_games)
    
    # Uncomment the following lines to test with RLAgent
    # from agents.RLAgent import RLAgent
    # agent1 = RLAgent(fog=True, faction=chess.WHITE)
    # agent2 = RLAgent(fog=True, faction=chess.BLACK)
    # test_win_rate(agent1, agent2, num_games)