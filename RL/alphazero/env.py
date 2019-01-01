import chess
from IPython.display import display, clear_output
import random
import time

def who(player):
    return "White" if player == chess.WHITE else "Black"

class ChessGame:
    """
    Base class where the chess game is played
    """
    
    def __init__(self, player1, player2):
        self.board = chess.Board()
        
        # Setting up the players
        self.white = player1
        self.black = player2
        self.white.board = self.board
        self.black.board = self.board
        self.white.my_color = chess.WHITE
        self.black.my_color = chess.BLACK
        
        # Game outcome
        self.winner = None
        self.over = False
        
        
    def white_move(self):
        """
        White makes a move according to their best move/model
        """
        assert self.board.turn == chess.WHITE
        self.white.get_move()

        
    def black_move(self):
        """
        Black makes a move according to their best move/model
        
        :param str action: black move (must be in UCI notation)
        """
        assert self.board.turn == chess.BLACK
        self.black.get_move()
        
    def display_board(self, use_svg = 1):
        """
        Method for use in iPython Notebook to visualize board
        """
        
        display(self.board)
        
    def play_game(self, pause=0.1, display = False):
        """
        Game play by the two agents comprising the game.
        
        :param float pause: how long to pause between displays (seconds)
        :param bool display: display the board after each move, verbose output
        """
        
        try:
            while not self.board.is_game_over(claim_draw=True):
                if self.board.turn == chess.WHITE:
                    self.white_move()
                else:
                    self.black_move()
                if display:
                    self.display_board()
                    time.sleep(pause)
                    clear_output(wait=True)
                    
                    
        except KeyboardInterrupt:
            msg = "Game interrupted!"
            return (None, msg)
        result = None
        if self.board.is_checkmate():
            msg = "checkmate: " + who(not self.board.turn) + " wins!"
            result = not self.board.turn
        elif self.board.is_stalemate():
            msg = "draw: stalemate"
        elif self.board.is_fivefold_repetition():
            msg = "draw: 5-fold repetition"
        elif self.board.is_insufficient_material():
            msg = "draw: insufficient material"
        elif self.board.can_claim_draw():
            msg = "draw: claim"
        if display:
            print(msg)
        return (result, msg)
        

