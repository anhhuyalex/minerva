import agents
import env

game = env.ChessGame(agents.RLPlayer(), agents.RLPlayer())
res, msg = game.play_game(display = 1)
print(game.board.ep_square)