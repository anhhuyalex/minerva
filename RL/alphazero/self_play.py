import env
import agents
import time
class SelfPlayGenerator:

    def __init__(self):
        self.white_NN, self.black_NN = agents.RLPlayer(), agents.RLPlayer()

    def start_game(self):
        game = env.ChessGame(self.white_NN, self.black_NN)
        white_data, black_data = [], []
        while not game.board.is_game_over(claim_draw=True):
            tic = time.time()
            white_board_input, white_normalized_move_counts = self.white_NN.get_move()
            toc = time.time()
            print("White moving took %f seconds" %(toc-tic))
            white_data.append([white_board_input, white_normalized_move_counts])
            tic = time.time()
            black_board_input, black_normalized_move_counts = self.black_NN.get_move()
            toc = time.time()
            print("Black moving took %f seconds" %(toc-tic))
            black_data.append([black_board_input, black_normalized_move_counts])
            print("Last 2 moves", game.board.move_stack[-2:])
            print(game.board)

        if game.board.result == "1-0":
            for move in white_data:
                move += [1]
            for move in black_data:
                move += [-1]
        elif game.board.result == "0-1":
            for move in white_data:
                move += [-1]
            for move in black_data:
                move += [1]
        else:
            for move in white_data:
                move += [0]
            for move in black_data:
                move += [0]

        return white_data + black_data

g = SelfPlayGenerator()
g.start_game()