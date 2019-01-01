def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array

def flipped_uci_labels():
    """
    Seems to somehow transform the labels used for describing the universal chess interface format, putting
    them into a returned list.
    :return:
    """
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in create_uci_labels()]

class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 3
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = True
    input_depth = 18

class TrainerConfig:
    def __init__(self):
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 5 # RAM explosion...
        self.vram_frac = 1.0
        self.batch_size = 384 # tune this to your gpu memory
        self.epoch_to_checkpoint = 1
        self.dataset_size = 100000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.loss_weights = [1.25, 1.0] # [policy, value] prevent value overfit in SL

class Config:
    """
    Config describing how to run the application
    Attributes (best guess so far):
        :ivar list(str) labels: labels to use for representing the game using UCI
        :ivar int n_lables: number of labels
        :ivar list(str) flipped_labels: some transformation of the labels
        :ivar int unflipped_index: idk
        :ivar Options opts: options to use to configure this config
        :ivar ResourceConfig resources: resources used by this config.
        :ivar ModelConfig mode: config for the model to use
        :ivar PlayConfig play: configuration for the playing of the game
        :ivar PlayDataConfig play_date: configuration for the saved data from playing
        :ivar TrainerConfig trainer: config for how training should go
        :ivar EvaluateConfig eval: config for how evaluation should be done
    """
    labels = create_uci_labels()
    n_labels = int(len(labels))
    flipped_labels = flipped_uci_labels()
    unflipped_index = None
    trainer = TrainerConfig()




#     def __init__(self, config_type="mini"):
#         """
#         :param str config_type: one of "mini", "normal", or "distributed", representing the set of
#             configs to use for all of the config attributes. Mini is a small version, normal is the
#             larger version, and distributed is a version which runs across multiple GPUs it seems
#         """
#         self.opts = Options()
#         self.resource = ResourceConfig()

#         if config_type == "mini":
#             import chess_zero.configs.mini as c
#         elif config_type == "normal":
#             import chess_zero.configs.normal as c
#         elif config_type == "distributed":
#             import chess_zero.configs.distributed as c
#         else:
#             raise RuntimeError(f"unknown config_type: {config_type}")
#         self.model = c.ModelConfig()
#         self.play = c.PlayConfig()
#         self.play_data = c.PlayDataConfig()
#         self.trainer = c.TrainerConfig()
#         self.eval = c.EvaluateConfig()
#         self.labels = Config.labels
#         self.n_labels = Config.n_labels
#         self.flipped_labels = Config.flipped_labels

#     @staticmethod
#     def flip_policy(pol):
#         """
#         :param pol policy to flip:
#         :return: the policy, flipped (for switching between black and white it seems)
#         """
#         return np.asarray([pol[ind] for ind in Config.unflipped_index])
