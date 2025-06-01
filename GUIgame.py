"""
Main driver file.
Handling user input.
Displaying current GameStatus object.
"""
import pygame as p
import sys
from multiprocessing import Process, Queue
import chess
from chess import Move
import random
from utils.BoardManipulation import *
import agents
import time

BOARD_WIDTH = BOARD_HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 250
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT
DIMENSION = 8
SQUARE_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

agent = agents.GreedyAgent()  # Initialize the agent

def loadImages():
    """
    Initialize a global directory of images.
    This will be called exactly once in the main.
    """
    pieces = ['P', 'R', 'N', 'B', 'K', 'Q', 'p', 'r', 'n', 'b', 'k', 'q']
    for piece in pieces:
        mod_name = 'w' + piece if (piece.upper() == piece) else piece
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + mod_name + ".png"), (SQUARE_SIZE, SQUARE_SIZE))


def main():
    """
    The main driver for our code.
    This will handle user input and updating the graphics.
    """
    p.init()
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    board = chess.Board()
    move_made = False  # flag variable for when a move is made
    loadImages()  # do this only once before while loop
    running = True
    square_selected = ()  # no square is selected initially, this will keep track of the last click of the user (tuple(row,col))
    player_clicks = []  # this will keep track of player clicks (two tuples)
    game_over = False
    ai_thinking = False
    move_undone = False
    move_finder_process = None
    move_log_font = p.font.SysFont("Arial", 14, False, False)
    player_one = True  # if a human is playing white, then this will be True, else False
    player_two = False  # if a hyman is playing white, then this will be True, else False

    while running:
        human_turn = (board.turn == chess.WHITE and player_one) or (board.turn == chess.BLACK and player_two)
        for e in p.event.get():
            if e.type == p.QUIT:
                p.quit()
                sys.exit()
            # mouse handler
            elif e.type == p.MOUSEBUTTONDOWN:
                print(game_over)
                if not game_over:
                    location = p.mouse.get_pos()  # (x, y) location of the mouse
                    col = location[0] // SQUARE_SIZE
                    row = location[1] // SQUARE_SIZE
                    if square_selected == (8 - row, col) or col >= 8:  # user clicked the same square twice
                        square_selected = ()  # deselect
                        player_clicks = []  # clear clicks
                    else:
                        square_selected = (8 - row, col)
                        player_clicks.append(square_selected)  # append for both 1st and 2nd click

                    if len(player_clicks) == 2 and human_turn:  # after 2nd click
                        
                        src = (player_clicks[0][0], chr(ord('a') + player_clicks[0][1]))
                        dest = (player_clicks[1][0], chr(ord('a') + player_clicks[1][1]))

                        move = Move.from_uci(f'{src[1]}{src[0]}{dest[1]}{dest[0]}')


                        # TODO block illegal move
                        player_clicks = [square_selected]
                        if(not board.pseudo_legal_moves.__contains__(move)) : continue

                        board.push(move)
                        square_selected = ()  # reset user clicks
                        player_clicks = []
                        

        # AI move finder
        if not game_over and not human_turn:
            time.sleep(0.5)
            board.push(agent.act(board))
            # board.push(
            #     random.choice(
            #         list(board.pseudo_legal_moves)))

        if move_made:
            move_made = False

        drawGameState(screen, board, square_selected, game_over)

        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            game_over = True
            drawGameState(screen, board, square_selected, game_over)
            if board.king(chess.WHITE) is None:
                drawEndGameText(screen, "Black wins")
            else:
                drawEndGameText(screen, "White wins")

        clock.tick(MAX_FPS)
        p.display.flip()

    

def drawGameState(screen, board : chess.Board, square_selected, game_over):
    """
    Responsible for all the graphics within current game state.
    """
    drawBoard(screen)  # draw squares on the board
    highlightSquares(screen, board, square_selected)
    drawPieces(screen, board_to_array(board))  # draw pieces on top of those squares
    drawFog(screen, board, game_over)


def drawBoard(screen):
    """
    Draw the squares on the board.
    The top left square is always light.
    """
    global colors
    colors = [p.Color("white"), p.Color("gray")]
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[((row + column) % 2)]
            p.draw.rect(screen, color, p.Rect(column * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def highlightSquares(screen, board : chess.Board, square_selected):
    """
    Highlight square selected and moves for piece selected.
    """
    if(len(board.move_stack) > 0):
        last_move = board.move_stack[-1]
        s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
        s.set_alpha(100)
        s.fill(p.Color('green'))

        scol = ord(last_move.uci()[2]) - ord('a')
        srow = 8 - int(last_move.uci()[3])

        screen.blit(s, (scol * SQUARE_SIZE, srow * SQUARE_SIZE))
    
    if square_selected != ():

        row, col = square_selected
        src_uci = f'{chr(ord("a") + col)}{row}'
        
        row = 8 - row
        
        pb : str = board_to_array(board)[row][col]
        if((board.turn == chess.WHITE and pb.isupper()) or
           (board.turn == chess.BLACK and not pb.isupper())):
            # highlight selected square
            s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(100)  # transparency value 0 -> transparent, 255 -> opaque
            s.fill(p.Color('blue'))
            screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
            # highlight moves from that square
            s.fill(p.Color('yellow'))

            for move in board.pseudo_legal_moves:

                if(not move.uci().startswith(src_uci)) : continue
            
                dcol = ord(move.uci()[2]) - ord('a')
                drow = 8 - int(move.uci()[3])

                screen.blit(s, (dcol * SQUARE_SIZE, drow * SQUARE_SIZE))


def drawPieces(screen, board):
    """
    Draw the pieces on the board using the current game_state.board
    """
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            piece = board[row][column]
            if piece != ".":
                screen.blit(IMAGES[piece], p.Rect(column * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def drawFog(screen, board : chess.Board, game_over):
    """
    Draw fog over squares that white cannot see, regardless of whose turn it is.
    """
    pb = board_to_array(board)
    fog_color = p.Color("#3a3a3a")

    visible_squares = set()

    # Reveal all white pieces
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = pb[row][col]
            if piece.isupper():
                visible_squares.add((row, col))

    tempboard = board.copy()
    if(tempboard.turn == chess.BLACK) : tempboard.push(Move.null())

    # Reveal valid move targets for white
    
    for move in tempboard.pseudo_legal_moves:
            
        dcol = ord(move.uci()[2]) - ord('a')
        drow = 8 - int(move.uci()[3])
        visible_squares.add((drow, dcol))


    # Apply fog to all other squares
    s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
    s.fill(fog_color)
    if game_over:
        for row in range(DIMENSION):
            for col in range(DIMENSION):
                if (row, col) not in visible_squares:
                    s.set_alpha(200)
                    screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
    else:
        for row in range(DIMENSION):
            for col in range(DIMENSION):
                if (row, col) not in visible_squares:
                    colors = [p.Color("white"), p.Color("gray")]
                    color = colors[((row + col) % 2)]
                    p.draw.rect(screen, color, p.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    s.set_alpha(225)
                    screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
    return

def drawMoveLog(screen, game_state, font, texting=False):
    """
    Draws the move log.

    """
    move_log_rect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color('black'), move_log_rect)
    move_log = game_state.move_log
    move_texts = []
    if texting:
        for i in range(0, len(move_log), 2):
            move_string = str(i // 2 + 1) + '. ' + str(move_log[i]) + " "
            if i + 1 < len(move_log):
                move_string += str(move_log[i + 1]) + "  "
            move_texts.append(move_string)

    moves_per_row = 3
    padding = 5
    line_spacing = 2
    text_y = padding
    for i in range(0, len(move_texts), moves_per_row):
        text = ""
        for j in range(moves_per_row):
            if i + j < len(move_texts):
                text += move_texts[i + j]

        text_object = font.render(text, True, p.Color('white'))
        text_location = move_log_rect.move(padding, text_y)
        screen.blit(text_object, text_location)
        text_y += text_object.get_height() + line_spacing


def drawEndGameText(screen, text):
    font = p.font.SysFont("Helvetica", 32, True, False)
    text_object = font.render(text, False, p.Color("gray"))
    text_location = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(BOARD_WIDTH / 2 - text_object.get_width() / 2,
                                                                 BOARD_HEIGHT / 2 - text_object.get_height() / 2)
    screen.blit(text_object, text_location)
    text_object = font.render(text, False, p.Color('black'))
    screen.blit(text_object, text_location.move(2, 2))

if __name__ == "__main__":
    main()
