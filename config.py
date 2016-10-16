config = {}
config['batch_size'] = 25  # number of samples taken per each update
config['hidden_size'] = 128  # number of hidden units per layer
config['max_morphemes_per_word'] = 8 # NOTE: This needs to be set depending on the dataset.
config['num_layers'] = 2
config['learning_rate'] = 0.001
config['learning_rate_decay'] = 0.97
config['decay_rate'] = 0.95
config['step_clipping'] = 1.0  # clip norm of gradients at this value
config['dropout'] = 0

config['nepochs'] = 10  # number of full passes through the training data
config['seq_length'] = 50  # number of chars in the sequence
config['train_size'] = 0.95  # fraction of data that goes into train set
config['save_path'] = '{0}_best.pkl'.format('lstm')  # path to best model file
config['load_path'] = '{0}_best.pkl'.format('lstm')  # start from a saved model file
config['last_path'] = '{0}_last.pkl'.format('lstm')  # path to save the model of the last iteration
