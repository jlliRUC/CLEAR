from data_augmentation import run_single
import argparse

parser = argparse.ArgumentParser(description="execute_single.py")
parser.add_argument("-dataset_name", help="Name of dataset")


if __name__ == '__main__':
    args = parser.parse_args()
    # single:
    print(f"single for {args.dataset_name}")
    params_single = []
    for name in ["distort", "downsampling", "interpolation"]:
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 'random']:
            run_single(args.dataset_name, {"name": name, "params": {"rate": i}})
