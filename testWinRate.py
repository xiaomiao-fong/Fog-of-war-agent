import pygame as p
import sys
from multiprocessing import Process, Queue
import chess
from chess import Move
import random
from utils.BoardManipulation import *
import agents
import time

#use random and greedy agent to play a game
def play_game(agent1, agent2, queue):
    board = chess.Board()
    agents = [agent1, agent2]
    current_agent_index = 0

    while not board.is_game_over():
        current_agent = agents[current_agent_index]
        move = current_agent.act(board)
        board.push(move)
        current_agent_index = 1 - current_agent_index  # Switch to the other agent

    queue.put(board.result())
def test_win_rate(agent1, agent2, num_games=100):
    white_wins = 0
    black_wins = 0
    draws = 0
    
    # Just try with single process for simplicity
    queue = Queue()
    for _ in range(num_games):
        p = Process(target=play_game, args=(agent1, agent2, queue))
        p.start()
        p.join()
        result = queue.get()
        
        if result == '1-0':
            white_wins += 1
        elif result == '0-1':
            black_wins += 1
        else:
            draws += 1

    print(f"White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
    print(f"Win rate for White: {white_wins / num_games * 100:.2f}%")
    print(f"Win rate for Black: {black_wins / num_games * 100:.2f}%")
    print(f"Draw rate: {draws / num_games * 100:.2f}%")
    if agent1.faction == chess.WHITE:
        print("White agent win rate: ", white_wins / num_games * 100)
        print("Black agent win rate: ", black_wins / num_games * 100)
    else:
        print("White agent win rate: ", black_wins / num_games * 100)
        print("Black agent win rate: ", white_wins / num_games * 100)
        print("Draw rate: ", draws / num_games * 100)
    return white_wins, black_wins, draws

if __name__ == "__main__":
    # Initialize agents
    agent1 = agents.GreedyAgent(faction=chess.WHITE)
    agent2 = agents.GreedyAgent(faction=chess.BLACK)

    # Test win rate
    num_games = 100
    test_win_rate(agent1, agent2, num_games)
    
    # Uncomment the following lines to test with RLAgent
    # from agents.RLAgent import RLAgent
    # agent1 = RLAgent(fog=True, faction=chess.WHITE)
    # agent2 = RLAgent(fog=True, faction=chess.BLACK)
    # test_win_rate(agent1, agent2, num_games)