import chess
import random
import time
import chess_parameters
import numpy as np

class HumanPlayer:
    """
    A human player can provide input to which move to take.
    """
    
    def __init__(self, notation = "UCI"):
        self.notation = notation
        self.board = None
        self.my_color = None
    
    def get_move(self):
        """
        User inputs move, must be a legal move in correct format.
        User gets 4 tries
        """
        for x in range(0, 4):  
            try:
                move = input("Please provide your move in %s notation (qqq to exit):" %self.notation)
                if move == "qqq":
                    raise KeyboardInterrupt
                if self.notation == "SAN":
                    self.board.push_san(move)
                else:
                    self.board.push_uci(move)
                return (move)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except ValueError:
                pass
                
        
        
    
class RandomPlayer:
    """
    Implements a random player that picks a random move 
    out of all possible moves.
    """
    def __init__(self):
        self.board = None
        self.my_color = None
    
    def get_move(self):
        move = random.choice(list(self.board.legal_moves))
        self.board.push(move)
        
class PieceValuePlayer:
    """
    Implements a player that selects the move that will maximize 
    their piece values on the board.
    
    The only improvement over a random player is that a PieceValuePlayer
    will capture a piece if they can.
    """
    def __init__(self):
        self.board = None
        self.my_color = None
    
    def get_move(self):
        move = self.pick_highest_value_move()
        self.board.push(move)
        
    def pick_highest_value_move(self):
        """
        For each move, evaluate the position after the move,
        then select the highest value move
        """
        # Evaluate move
        moves = list(self.board.legal_moves)
        for move in moves:
            hypothetical_board = self.board.copy()
            move.score = self.static_analysis(move, hypothetical_board, self.board.turn)

        # Select random move among the moves that do best
        best_move_score = max([move.score for move in moves])
        best_moves = [move for move in moves if move.score == best_move_score]
        random_best_move = random.choice(best_moves)
        return random_best_move
    
        
    def static_analysis(self, move, board, my_color):
        """
        Evaluate the board position according to point values 
        purely based on the pieces on the board
        
        Piece values are:
            Pawn: 1
            Bishop: 4
            Queen: 10
            Knight: 3
            Rook: 5
            
        :param chess.Move move: move to evaluate
        :param chess.Board board: board on which to evaluate move
        :param bool my_color: perspective with which to evaluate moves
        """
        board.push(move)
        score = 0
        for (piece, value) in [(chess.PAWN, 1), 
                           (chess.BISHOP, 4), 
                           (chess.QUEEN, 10), 
                           (chess.KNIGHT, 3),
                           (chess.ROOK, 5)]:
            score += len(board.pieces(piece, my_color)) * value
            score -= len(board.pieces(piece, not my_color)) * value
        score += 100 if board.is_checkmate() else 0
        return score
    
class MinimaxPlayer:
    """
    Implements a player that expands the game tree up to a certain depth
    and selects the minimax move.
    
    The minimax move is the best case scenario assuming the a worst case 
    opponent. Assuming the opponent is as strong as possible and will respond 
    with their best move, we select the move that will maximize our payoff
    in this worst case scenario.
    """
    def __init__(self, depth = 2):
        """
        Class constructor for minimax player.
        
        
        """
        self.board = None
        self.max_depth = depth
        self.my_color = None
    
    def get_move(self):
        move = self.minimax(self.max_depth)
        self.board.push(move)
        
    
    def minimax(self, depth, simulate_opponent = False):
        """
        Recursive function that expands every 
        """
        moves = list(self.board.legal_moves) 
        # If no more legal moves, then game has ended
        if len(moves) == 0:
            moves = None
            print (self.board.result)
            return self.static_analysis(None, self.board, self.my_color)
            
        # At leaf node, evaluate board position
        if depth == 1: 
            leaf_scores = [self.static_analysis(move, self.board, self.my_color) for move in moves]
            if simulate_opponent:
                return min(leaf_scores)
            else:
                return max(leaf_scores)
        
        # If simulating opponent, opponent tries to minimize my score
        if simulate_opponent:
            for move in moves:
                self.board.push(move)
                move.score = self.minimax(depth - 1, simulate_opponent = not simulate_opponent) 
                #the better the opponent scores, the worse we score
                self.board.pop()
            return min([move.score for move in moves])
        else:
        # If simulating my moves, I try to maximize my own score
            for move in moves:
                self.board.push(move)
                move.score = self.minimax(depth - 1, simulate_opponent = not simulate_opponent) 
                print ("Move:", self.board.uci(move), "score", move.score)
                #the better the opponent scores, the worse we score
                self.board.pop()
            if depth < self.max_depth:
                return max([move.score for move in moves])

        best_move_score = max([move.score for move in moves])
        best_moves = [move for move in moves if move.score == best_move_score]
        random_best_move = random.choice(best_moves)
        return random_best_move
    
        
    def static_analysis(self, move, board, my_color):
        """
        Evaluate the board position according to point values previously
        described
        
        Piece values are:
            Pawn: 1
            Bishop: 4
            Queen: 10
            Knight: 3
            Rook: 5
        """
        # Return score if the game has ended
        if move == None:
            if board.result == "1-0":
                score = 100 if my_color == 1 else -100
            elif board.result == "0-1":
                score = -100 if my_color == 1 else 100
            else:
                score = 0 
            return score
        else:
            score = 0
                
        board.push(move)
        
        for (piece, value) in [(chess.PAWN, 100), 
                           (chess.BISHOP, 330), 
                           (chess.QUEEN, 900), 
                           (chess.KNIGHT, 320),
                           (chess.ROOK, 500)]:
            my_piece_position = board.pieces(piece, my_color)
            score += len(my_piece_position) * value
            for position in my_piece_position:
                score += chess_parameters.POSITION_dictionary[piece][position]
            opponent_piece_position = board.pieces(piece, not my_color)
            score -= len(opponent_piece_position) * value
            for position in opponent_piece_position:
                score -= chess_parameters.POSITION_dictionary[piece][position]
                
        # Evaluate king safety/activity depending on mid/end game
        my_king = list(board.pieces(chess.KING, my_color))[0]
        opponent_king = list(board.pieces(chess.KING, not my_color))[0]
        if self.board.fullmove_number < 50:
            score += chess_parameters.POSITION_dictionary[chess.KING][0][my_king]
            score -= chess_parameters.POSITION_dictionary[chess.KING][0][opponent_king]
        else:
            score += chess_parameters.POSITION_dictionary[chess.KING][1][my_king]
            score -= chess_parameters.POSITION_dictionary[chess.KING][1][opponent_king]
        
    
        score += 20000 if board.is_checkmate() else 0
        board.pop()
        return score
    
class AlphaBetaPlayer(MinimaxPlayer):
    """
    Implements a player that expands the game tree up to a certain depth,
    prunes the tree according to the alpha-beta pruning algorith,
    and selects the minimax move.
    
    The minimax move is the best case scenario assuming the a worst case 
    opponent. Assuming the opponent is as strong as possible and will respond 
    with their best move, we select the move that will maximize our payoff
    in this worst case scenario.
    
    The alpha-beta pruning algorithm avoids searching through nodes that 
    are guaranteed to violate the assumption that the opponent is as 
    strong as possible, allowing a more in-depth search.
    """
    
    def get_move(self):
        move = self.alphabeta(move = 0, depth = self.max_depth)
        self.board.push(move)
    
    def alphabeta(self, move, depth, simulate_opponent = False, 
                alpha = -np.inf, beta = np.inf):
        """
        Recursive function that expands the minimax game tree and prunes
        it according to the alpha-beta pruning algorithm
        """
        
        # At leaf node, evaluate board position
        if depth == 0: 
            return self.static_analysis(move, self.board, self.my_color) 
        
        moves = list(self.board.legal_moves) 
        
        # Heuristics for optimal ordering for alpha-beta pruning
        moves = sorted(moves, key = self.ordering_heuristics, reverse=True)
        
        # If no more legal moves, then game has ended. Evaluate leaf node
        if len(moves) == 0:
            return self.static_analysis(None, self.board, self.my_color)
            
        # If simulating opponent, opponent tries to minimize my score
        if simulate_opponent:
            best_move_score = np.inf
            for move in moves:
                self.board.push(move)
                move.score = self.alphabeta(move, depth - 1, not simulate_opponent,
                                         alpha, beta) 
                best_move_score = min(best_move_score, move.score)
                beta = min(beta, best_move_score)
                self.board.pop()
                #the better the opponent scores, the worse we score
                if beta < alpha:
                    break
            return best_move_score
        else:
        # If simulating my moves, I try to maximize my own score
            best_move_score = -np.inf
            for move in moves:
                self.board.push(move)
                move.score = self.alphabeta(move, depth - 1, not simulate_opponent,
                                         alpha, beta) 
                best_move_score = max(best_move_score, move.score)
                alpha = max(alpha, best_move_score)
                self.board.pop()

                if beta <= alpha:
                    break

            if depth < self.max_depth:
                return best_move_score

        # Select a best move from a list of best moves
        best_move_score = max([move.score for move in moves])
        best_moves = [move for move in moves if move.score == best_move_score]
        random_best_move = random.choice(best_moves)
        
        return random_best_move
    
        
    
    
    def ordering_heuristics(self, move):
        move_order = self.board.is_capture(move)
        move_order += self.board.is_attacked_by(not self.board.turn, move.from_square)
        
        return move_order
    
    
    
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from multiprocessing import connection, Pipe
from threading import Thread

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np 
import neural_network_config


    
class NodeStatistics:
    """
    Holds the stats needed for the AGZ MCTS algorithm for a specific action taken from a specific state.
    Attributes:
        :ivar int n: number of visits to this action by the algorithm
        :ivar float w: every time a child of this action is visited by the algorithm,
            this accumulates the value (calculated from the value network) of that child. This is modified
            by a virtual loss which encourages threads to explore different nodes.
        :ivar float q: mean action value (total value from all visits to actions
            AFTER this action, divided by the total number of visits to this action)
            i.e. it's just w / n.
        :ivar float p: prior probability of taking this action, given
            by the policy network.
            
    ## EDIT
    """
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0
        
class RLPlayer:
    """
    Implements a player that uses neural networks to play chess.
    
    Attributes:
    """
    def __init__(self, config = None):
        """
        Class constructor for minimax player.
        
        
        """
        self.board = None
        self.my_color = None
        self.model = None
        if config == None:
            self.config_model = neural_network_config.ModelConfig()
            self.config = neural_network_config.Config()
        self.node_lock = defaultdict(Lock)
        self.game_tree = {}

        # Set up multiprocessing for speed 
        self.feed_input, self.return_policy_value = self.create_pipes()
        
        # Build a model
        self.build_model()
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised 
        self.model.compile(optimizer=Adam(), loss=losses, loss_weights=self.config.trainer.loss_weights)

        # Dictionary to facilitate converting from network outputs to move
        self.move_code = {i: chess.Move.from_uci(move) 
                          for move, i in zip(neural_network_config.create_uci_labels(), 
                          range(len(neural_network_config.create_uci_labels())))}
        self.move_lookup = {chess.Move.from_uci(move): i
                          for move, i in zip(neural_network_config.create_uci_labels(), 
                          range(len(neural_network_config.create_uci_labels())))}
        
        # Start a thread to listen on the pipe and make predictions
        self.prediction_worker = Thread(target=self._predict_batch_worker, name = "prediction_worker")
        self.prediction_worker.daemon = True
        self.prediction_worker.start()

    def create_pipes(self):
        self.feed_input, self.return_policy_value = [], []
        for thread in range(30):
            me, you = Pipe()
            self.feed_input.append(me)
            self.return_policy_value.append(you)
        
        return self.feed_input, self.return_policy_value
    
    def get_move(self):
        
        # Perform Monte-Carlo Tree Search (updating internal variables)
        self.MCTS()
        
        # Choose the most visited node (highest exponentiated visited)
        state = state_key(self.board)

        candidate_moves = self.game_tree[state]['action']

        # Temperature controls exploration depending on stage of game
        board_input, move_counts = self.save_move(candidate_moves)
        if self.board.fullmove_number < 45:
            if self.board.fullmove_number < 30:
                temperature = 0.95 ** self.board.fullmove_number
            else:
                temperature = 0.1
            exp_move_counts = np.power(move_counts, 1./temperature)
            exp_move_counts /= np.sum(exp_move_counts)
        else:
            exp_move_counts = np.zeros_like(move_counts)
            exp_move_counts[np.argmax(move_counts)] = 1
        
        move = np.random.choice([move for move in candidate_moves], p = exp_move_counts)
        self.board.push(move)

        # Return data (useful only for self-play generation)
        normalized_move_counts = move_counts / np.sum(move_counts)
        return board_input, normalized_move_counts
        
    
        
    def build_model(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        mc = self.config_model
        in_x = x = Input((18, 8, 8))

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x
        
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        mc = self.config_model
        in_x = x
        res_name = "res"+str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    def visualize_model(self):
        """
        Print out model summary (contains layer names, shape of input, 
        number of parameters, and connection to)
        """
        self.model.summary()
        
    
        
    
                
    def MCTS(self):
        """
        Using 30 workers (max_workers=self.play_config.search_threads)
        self.play_config.simulation_num_per_move = 800
        """
        futures = []
        
        with ThreadPoolExecutor(max_workers = 30) as executor:
            for _ in range(800):
            # self.select_move(board=self.board.copy(),is_root_node=True)
                future = executor.submit(self.select_move,board=self.board.copy(),is_root_node=True)
            # if future.exception():
                    # raise ValueError
                # The board is copied so I don't need to pop the move 

#         vals = [f.result() for f in futures]
        
    def select_move(self, board, is_root_node=False):
        """
        They use virtual_loss
        """
        # print (self.node_lock)
        state = state_key(board)

        with self.node_lock[state]:
            if state not in self.game_tree:
                # print(state)
                policy, value = self.forward_pass(board)
                # print(policy, value)
            # if state not in self.game_tree:
                # print ("I'm evaluating leaf", board.move_stack)
            #     
                self.game_tree[state] = {}
                self.game_tree[state]['policy'] = policy
                self.game_tree[state]['action'] = defaultdict(NodeStatistics)
                self.game_tree[state]['total_visits'] = 1
                # print (self.game_tree)
            #     # I must have visited once before to call best_q_move method
                return value
                
        action = self.best_q_move(board, is_root_node)
        # print (action)
        board.push(action)
            
        # Simulate enemy_move
        enemy_value = self.select_move(board)
        value = -enemy_value
            
        actions = self.game_tree[state]['action']
        with self.node_lock[state]:
            self.game_tree[state]['total_visits'] += 1
            actions[action].n += 1
            actions[action].w += value
            actions[action].q = actions[action].w / actions[action].n
            
        return value
        
    def best_q_move(self, board, is_root_node):
        """
        c_puct = 1.5
        """
        # print ("Hi")
        state = state_key(board)

        policy = self.game_tree[state]['policy']
        actions = self.game_tree[state]['action']
        unnormalized_prior = [policy[self.move_lookup[move]] for move in board.legal_moves]
        
        # print (unnormalized_prior)
        prior = unnormalized_prior / sum(unnormalized_prior)
        sqrt_total_visits = np.sqrt(self.game_tree[state]['total_visits'])

        
        c_puct = 1.5
        dirichlet_alpha = 0.3
        noise_eps = 0.25
        best_q = -np.inf
        best_move = None
        num_legal_moves = len(list(board.legal_moves))

        if is_root_node:
            dirichlet_noise = np.random.dirichlet([dirichlet_alpha] * num_legal_moves)
        for index, move in enumerate(board.legal_moves):
            candidate_q = (actions[move].q + 
                       c_puct * prior[index] * sqrt_total_visits / (1 + actions[move].n))
            if is_root_node: #add noise for exploration
                candidate_q = ((1 - noise_eps) * candidate_q + 
                noise_eps * dirichlet_noise[index])
            if (best_q < candidate_q):
                best_q = candidate_q
                best_move = move
                
        return best_move   
        
        

    def forward_pass(self, board):
        input_planes = self.board_to_input(board, board.turn)
        # print (input_planes)
        input_pipe = self.feed_input.pop()
        input_pipe.send(input_planes)
        policy, value = input_pipe.recv()
        self.feed_input.append(input_pipe)
        return policy, value

                
    def _predict_batch_worker(self):
        """
        Thread worker which listens on each pipe in self.pipes for an observation, and then outputs
        the predictions for the policy and value networks when the observations come in. Repeats.
        
        ## CITE
        """
        while True:
            ready = connection.wait(self.return_policy_value,timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            # print (data.shape)
            
            policy_array, value_array = self.model.predict_on_batch(data)
            # print (policy_array, value_array)
            for pipe, policy, value in zip(result_pipes, policy_array, value_array):
                pipe.send((policy, float(value)))
        
    def board_to_input(self, board, my_color = None):
        """
        FIX YOUR COLOR PROBLEM: ASSUME THAT THE NEURAL NETWORK RECEIVES THE INPUT FROM WHITE'S PERSPECTIVE
    
        Input: 18 planes of size (8,8) representing the entire board
        Boolean values: first 6 planes represent my pawn, knight, bishop, rook, queen, king
        Next 6 planes represent opponent's pieces (in the same order)
        Next 4 planes represent my king queen castling and opponents king queen castling
        Next plane represents half move clock (50 move without pawn advance or piece capture is a draw)
        Next plane represents the en passant square (if available)
        """
        

        if my_color == None:
            my_color = self.my_color
        pieces_planes = np.zeros(shape=(12, 8, 8), dtype=np.float32)
        board_colors = [not my_color, my_color]
        en_passant = np.zeros((8, 8), dtype=np.float32)
        
#         print (board_colors)
        if my_color == 0:
            for my_board, color in enumerate(board_colors):
                for piece in range(1, 7):
                    my_piece_position = board.pieces(piece, color)
                    rank, file = np.array([[(int(i / 8)) for i in list(my_piece_position) ], 
                                     [(7-(i % 8)) for i in list(my_piece_position) ]])
                    pieces_planes[(piece - 1) + (my_board + 1) % 2 * 6, rank, file] = 1
            if board.ep_square != None:
                en_passant[int(board.ep_square / 8), 7 - (board.ep_square % 8)] = 1
        else:
            # print ("Yo my color is", my_color)
            for my_board, color in enumerate(board_colors):
                for piece in range(1, 7):
                    my_piece_position = board.pieces(piece, color)
                    rank, file = np.array([[(7 - int(i / 8)) for i in list(my_piece_position) ], 
                                     [(i % 8) for i in list(my_piece_position) ]])
                    pieces_planes[(piece - 1) + (my_board + 1) % 2 * 6, rank, file] = 1
            if board.ep_square != None:
                en_passant[7 - int(board.ep_square / 8), (board.ep_square % 8)] = 1
            
        # print("Hi")
        
        
        auxiliary_planes = np.array([np.full((8, 8), board.has_kingside_castling_rights(my_color), dtype=np.float32),
                        np.full((8, 8), board.has_queenside_castling_rights(my_color), dtype=np.float32),
                        np.full((8, 8), board.has_kingside_castling_rights(not self.my_color), dtype=np.float32),
                        np.full((8, 8), board.has_queenside_castling_rights(not my_color), dtype=np.float32),
                        np.full((8, 8), board.halfmove_clock, dtype=np.float32),
                        en_passant])
        
        # print (np.vstack((pieces_planes, auxiliary_planes)))
        return (np.vstack((pieces_planes, auxiliary_planes)))

    
    def save_move(self, candidate_moves):
        """
        Used by the self-play generator to generate move-policy data
        """
        board_input = self.board_to_input(self.board.copy())
        move_counts = np.array([candidate_moves[move].n for move in candidate_moves])     
        return board_input, move_counts

def state_key(board) -> str:
    """
    :param ChessEnv env: env to encode
    :return str: a str representation of the game state
    """
    fen = board.fen().rsplit(' ', 1) # drop the move clock
    return fen[0]