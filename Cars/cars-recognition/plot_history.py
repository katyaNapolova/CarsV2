import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.use('Agg')


def get_args():
    parser = argparse.ArgumentParser(description="This script shows training graph from history file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input history h5 file")
    parser.add_argument("--start", type=int, default=0, help="draw graph from point in history")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    start = args.start
    input_path = args.input
    input_dir = os.path.dirname(input_path)

    history = pd.read_hdf(input_path, "history")

    # list all data in history
    print(history.keys())
    # summarize history for accuracy
    plt.plot(history['acc'][start:])
    plt.plot(history['val_acc'][start:])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(input_dir, "accuracy.png"))

    # summarize history for loss
    plt.plot(history['loss'][start:])
    plt.plot(history['val_loss'][start:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(input_dir, "loss.png"))


if __name__ == '__main__':
    main()
