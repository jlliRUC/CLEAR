from STToken import token_grid
import argparse
import os

parser = argparse.ArgumentParser(description="execute_grid.py")
parser.add_argument("-dataset_name", help="Name of dataset")
parser.add_argument("-cell_size")
parser.add_argument("-minfreq")


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.dataset_name)
    encoded_folder = f"../data/{args.dataset_name}/token"
    if not os.path.exists(f"..data/{args.dataset_name}/token"):
        os.mkdir(encoded_folder)
    token_grid(args.dataset_name, encoded_folder, args.cell_size, args.minfreq)