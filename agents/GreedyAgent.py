import chess
from chess import Move
from .BaseAgent import BaseAgent
import random

class GreedyAgent(BaseAgent):
    def __init__(self):
        super().__init__("GreedyAgent")

    def get_black_visible_squares(self, board):
        # Get all squares that are visible to the black pieces
        black_visible_squares = set()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == chess.BLACK:
                black_visible_squares.add(square)
                # add pawn moves
                if piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                    # Pawns attack diagonally forward
                    if square + 7 in chess.SQUARES:
                        black_visible_squares.add(square + 7)
                    if square + 9 in chess.SQUARES:
                        black_visible_squares.add(square + 9)
                # other pieces can see all squares they can move to
                elif piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    for move in board.legal_moves:
                        if move.from_square == square and move.to_square not in black_visible_squares:
                            black_visible_squares.add(move.to_square)
        return black_visible_squares



    def act(self, board: chess.Board) -> Move:
        # Find the best move based on the evaluation of the board
        best_move = set
        best_value = float('inf')
        
        for move in board.legal_moves:
            # print("move: " + str(move), end='\t;')
            board.push(move)
            value = self.evaluate_board(board)
            board.pop()
            
            if abs(value - best_value) < 1e-5:  # Check for equality with a tolerance
                best_move.add(move)
            elif value < best_value:
                best_move = {move}
                best_value = value
                
        return random.choice(list(best_move)) if best_move else None

    def evaluate_board(self, board: chess.Board) -> int:
        # evaluate the best move in black vision
        piece_values = {
            chess.PAWN: 10,
            chess.KNIGHT: 30,
            chess.BISHOP: 30,
            chess.ROOK: 50,
            chess.QUEEN: 90,
            chess.KING: 999  # King is invaluable
        }
        value = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                value += piece_values.get(piece.piece_type, 0)
        
        # add value for getting vision of black pieces
        visi=0
        black_visible_squares = self.get_black_visible_squares(board)
        for square in black_visible_squares:
            value -= 0.1
        # print("value: " + str(value))
        return value