import getopt
import sys
from forecast import Polenn


def print_help(save_freq, epochs, learning_rate, batch_size):
    print('Arguments:')
    print('-h to print help')
    print('-l to load model from file')
    print('-s to set save frequency(default = {})'.format(save_freq))
    print('-n to set number of epochs(default = {})'.format(epochs))
    print('-r to set learning rate(default = {})'.format(learning_rate))
    print('-b to set batch size(default = {})'.format(batch_size))


def main(argv):
    # Default values
    save_freq = 3
    epochs = 3
    load = False
    learning_rate = 0.005
    batch_size = 256

    try:
        opts, args = getopt.getopt(argv, 'hls:n:r:b:')
    except getopt.GetoptError:
        print_help(save_freq, epochs, learning_rate, batch_size)
        sys.exit(2)

    for arg, value in opts:
        # Print help
        if arg == '-h':
            print_help(save_freq, epochs, learning_rate, batch_size)
        # Load model?
        if arg == '-l':
            load = True
        # Save frequency
        if arg == '-s':
            save_freq = int(value)
        # Number of epochs
        if arg == '-n':
            epochs = int(value)
        # Learning rate
        if arg == '-r':
            learning_rate = float(value)
        # Batch size
        if arg == '-b':
            batch_size = int(value)

    model = Polenn()
    if load:
        model.load()
    else:
        model.create()

    model.compile(learning_rate=learning_rate)
    model.train(epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main(sys.argv[1:])
