import chess
from chess import Move
from .BaseAgent import BaseAgent
import random

class GreedyAgent(BaseAgent):
    def __init__(self, faction):
        super().__init__("GreedyAgent")
        self.faction = chess.WHITE if faction == "WHITE" else chess.BLACK
        self.opponent_faction = chess.BLACK if self.faction == chess.WHITE else chess.WHITE

    def get_black_visible_squares(self, board):
        # Get all squares that are visible to the black pieces
        black_visible_squares = set()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == self.faction:
                black_visible_squares.add(square)
                # add pawn moves
                if piece.piece_type == chess.PAWN and piece.color == self.faction:
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
        
        for move in board.pseudo_legal_moves:
            # print("move: " + str(move), end='\t;')
            board.push(move)
            nowd = board.copy()
            nowd.pop()  # Undo the move to evaluate the next one
            value = self.evaluate_board(board, nowd)
            board.pop()  # Undo the move to restore the board state
            
            if abs(value - best_value) < 1e-5:  # Check for equality with a tolerance
                best_move.add(move)
            elif value < best_value:
                best_move = {move}
                best_value = value
                
        return random.choice(list(best_move)) if best_move else random.choice(list(board.pseudo_legal_moves))

    def evaluate_board(self, board: chess.Board, vision) -> int:
        # check if the white king is in check
        know_black_checked = self.know_black_checked_in_vision(board)
        if know_black_checked:
            return float('inf')
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
            if piece and piece.color == self.opponent_faction:
                value += piece_values.get(piece.piece_type, 0)
        
        # add value for getting vision of black pieces
        black_visible_squares = self.get_black_visible_squares(board)
        for square in black_visible_squares:
            value -= 0.1
        # print("value: " + str(value))
        return value
    
    def know_black_checked_in_vision(self, board: chess.Board) -> bool:
        # 1. get all squares that are visible to the black pieces
        black_visible_squares = self.get_black_visible_squares(board)
        # 2. check if the white king is in check
        white_king_square = board.king(self.opponent_faction)
        # 3. if the white king is in check, check if the square is visible to the black pieces
        if white_king_square in black_visible_squares:
            return False
        # 4. if the white king is not in check, check if black king is in check by visiable white pieces
        black_king_square = board.king(self.faction)
        for square in black_visible_squares:
            piece = board.piece_at(square)
            if piece and piece.color == self.opponent_faction:
                if board.is_attacked_by(self.opponent_faction, black_king_square):
                    return True
