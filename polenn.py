import getopt
import sys
from model import Polenn


def train_help(save_freq, cycles, learning_rate, batch_size):
    print('Arguments:')
    print('-h to print help')
    print('-l to load model from file')
    print('-c to set cycle length(epochs before saving)(default = {})'.format(save_freq))
    print('-n to set number of cycles(default = {})'.format(cycles))
    print('-r to set learning rate(default = {})'.format(learning_rate))
    print('-b to set batch size(default = {})'.format(batch_size))


def plot_help(rows):
    print('Arguments:')
    print('-h to print help')
    print('-r to set number of rows(default = {})'.format(rows))


# Parse parameters for plotting
def plot_args(argv):
    rows = 2
    try:
        opts, args = getopt.getopt(argv, 'hr:')
    except getopt.GetoptError:
        plot_help(rows)

    for arg, value in opts:
        if arg == '-h':
            plot_help(rows)
        if arg == '-r':
            rows = int(value)

    model = Polenn()
    model.load()
    model.plot_predictions(rows)


# Parse parameters for training
def train_args(argv):
    # Default values
    save_freq = 3
    cycles = 3
    load = False
    learning_rate = 0.005
    batch_size = 256

    try:
        opts, args = getopt.getopt(argv, 'hlc:n:r:b:')
    except getopt.GetoptError:
        train_help(save_freq, cycles, learning_rate, batch_size)
        sys.exit(2)

    for arg, value in opts:
        # Print help
        if arg == '-h':
            train_help(save_freq, cycles, learning_rate, batch_size)
        # Load model?
        if arg == '-l':
            load = True
        # Save frequency
        if arg == '-c':
            save_freq = int(value)
        # Number of epochs
        if arg == '-n':
            cycles = int(value)
        # Learning rate
        if arg == '-r':
            learning_rate = float(value)
        # Batch size
        if arg == '-b':
            batch_size = int(value)

    train(save_freq, cycles, load, learning_rate, batch_size)


# Train the model
def train(save_freq, cycles, load, learning_rate, batch_size):
    model = Polenn()
    if load:
        model.load()
    else:
        model.create()

    model.compile(learning_rate=learning_rate)

    for e in range(cycles):
        model.train(epochs=save_freq, batch_size=batch_size)
        model.save()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train_args(sys.argv[2:])
    elif sys.argv[1] == 'plot':
        plot_args(sys.argv[2:])
