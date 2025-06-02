import chess
from chess import Move
from .BaseAgent import BaseAgent
import random

class GreedyAgent(BaseAgent):
    def __init__(self, faction):
        super().__init__("GreedyAgent")
        self.faction = chess.WHITE if faction == "WHITE" else chess.BLACK
        self.opponent_faction = chess.BLACK if self.faction == chess.WHITE else chess.WHITE

    def get_self_visible_squares(self, board):
        # Get all squares that are visible to the self pieces
        self_visible_squares = set()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == self.faction:
                self_visible_squares.add(square)
                # add pawn moves
                if piece.piece_type == chess.PAWN and piece.color == self.faction:
                    # Pawns attack diagonally forward
                    if square + 7 in chess.SQUARES:
                        self_visible_squares.add(square + 7)
                    if square + 9 in chess.SQUARES:
                        self_visible_squares.add(square + 9)
                # other pieces can see all squares they can move to
                elif piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    for move in board.legal_moves:
                        if move.from_square == square and move.to_square not in self_visible_squares:
                            self_visible_squares.add(move.to_square)
        return self_visible_squares



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
        know_self_checked = self.know_self_checked_in_vision(board)
        if know_self_checked:
            return float('inf')
        # evaluate the best move in self vision
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
        
        # add value for getting vision of self pieces
        self_visible_squares = self.get_self_visible_squares(board)
        for square in self_visible_squares:
            value -= 0.1
        # print("value: " + str(value))
        return value
    
    def know_self_checked_in_vision(self, board: chess.Board) -> bool:
        # 1. get all squares that are visible to the self pieces
        self_visible_squares = self.get_self_visible_squares(board)
        # 2. check if the oppo king is in check
        oppo_king_square = board.king(self.opponent_faction)
        # 3. if the oppo king is in check, check if the square is visible to the self pieces
        if oppo_king_square in self_visible_squares:
            return False
        # 4. if the oppo king is not in check, check if self king is in check by visiable white pieces
        self_king_square = board.king(self.faction)
        for square in self_visible_squares:
            piece = board.piece_at(square)
            if piece and piece.color == self.opponent_faction:
                # check if the piece can attack the self king
                if self.can_attack_king(piece, square, self_king_square, board):
                    return True
        return False
    def can_attack_king(self, piece, from_square, king_square, board):
        if from_square == king_square:
            return False
        # Check if the piece can attack the king square
        if piece.piece_type == chess.PAWN:
            # Pawns attack diagonally forward
            if from_square + 7 == king_square or from_square + 9 == king_square:
                return True
        elif piece.piece_type == chess.KNIGHT:
            # Knights have specific L-shaped moves
            knight_moves = [from_square + 15, from_square + 17, from_square - 15, from_square - 17,
                            from_square + 6, from_square - 6, from_square + 10, from_square - 10]
            if king_square in knight_moves:
                return True
        elif piece.piece_type == chess.BISHOP:
            # Bishops move diagonally
            dx = abs((king_square % 8) - (from_square % 8))
            dy = abs((king_square // 8) - (from_square // 8))
            if dx == dy:
                # Check if the path is clear
                step_x = 1 if (king_square % 8) > (from_square % 8) else -1
                step_y = 1 if (king_square // 8) > (from_square // 8) else -1
                for i in range(1, dx):
                    if board.piece_at(from_square + i * step_x + i * step_y * 8):
                        return False
                return True
        elif piece.piece_type == chess.ROOK:
            # Rooks move horizontally or vertically
            if from_square % 8 == king_square % 8 or from_square // 8 == king_square // 8:
                # Check if the path is clear
                step = 1 if (king_square % 8) > (from_square % 8) else -1
                for i in range(1, abs((king_square % 8) - (from_square % 8))):
                    if board.piece_at(from_square + i * step):
                        return False
                return True
            if from_square // 8 == king_square // 8:
                # Check if the path is clear vertically
                step = 1 if (king_square // 8) > (from_square // 8) else -1
                for i in range(1, abs((king_square // 8) - (from_square // 8))):
                    if board.piece_at(from_square + i * step * 8):
                        return False
                return True
        elif piece.piece_type == chess.QUEEN:
            # Queens move like both rooks and bishops
            if (from_square % 8 == king_square % 8 or from_square // 8 == king_square // 8 or
                abs((king_square % 8) - (from_square % 8)) == abs((king_square // 8) - (from_square // 8))):
                # Check if the path is clear
                if from_square % 8 == king_square % 8:
                    step = 1 if (king_square // 8) > (from_square // 8) else -1
                    for i in range(1, abs((king_square // 8) - (from_square // 8))):
                        if board.piece_at(from_square + i * step * 8):
                            return False
                    return True
                elif from_square // 8 == king_square // 8:
                    step = 1 if (king_square % 8) > (from_square % 8) else -1
                    for i in range(1, abs((king_square % 8) - (from_square % 8))):
                        if board.piece_at(from_square + i * step):
                            return False
                    return True
                else:
                    dx = abs((king_square % 8) - (from_square % 8))
                    dy = abs((king_square // 8) - (from_square // 8))
                    step_x = 1 if (king_square % 8) > (from_square % 8) else -1
                    step_y = 1 if (king_square // 8) > (from_square // 8) else -1
                    for i in range(1, dx):
                        if board.piece_at(from_square + i * step_x + i * step_y * 8):
                            return False
                    return True
        return False
