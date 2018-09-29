import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--file1', type=str)
arg_parser.add_argument('--file2', type=str)

args = arg_parser.parse_args()

with open(args.file1, 'r') as f1, open(args.fil2, 'r') as f2:
  for line1, line2 in zip(f1, f2):



