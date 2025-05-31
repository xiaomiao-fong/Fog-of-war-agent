import chess
import json

def board_to_array(board : chess.Board) -> list[list[str]]:
    return [r.split(' ') for r in str(board).split('\n')]

def array_to_board(board : list[list[str]]) -> chess.Board:
    
    fen = ""
    for r in board:
        sp_cnt = 0

        for c in r:

            if c == ".":
                sp_cnt += 1
                continue

            fen += c
            if(sp_cnt > 0) : fen += str(sp_cnt)
            
            sp_cnt = 0
        
        if sp_cnt > 0 : fen += str(sp_cnt)
        fen += "/"
    fen = fen[0:-1]
            
    return chess.Board(fen)

def fog_board(board : chess.Board) -> list[list[str]]:
    
    board_arr = board_to_array(board)
    board_mask = json.loads(json.dumps([[False] * 8] * 8))
    # "-" for fog

    for r in range(len(board_arr)):
        for c in range(len(board_arr)):
            if(board_arr[r][c].isupper()) : board_mask[r][c] = 1

    for move in board.pseudo_legal_moves:

        dcol = ord(move.uci()[2]) - ord('a')
        drow = int(move.uci()[3])
        board_mask[8 - drow][dcol] = 1
    
    for r in range(len(board_arr)):
        for c in range(len(board_arr)):
            if not board_mask[r][c]: 
                board_arr[r][c] = "-"

    return board_arr



