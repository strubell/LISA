import argparse
import nltk
from nltk.tokenize.nist import NISTTokenizer
nltk.download('perluniprops')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--file1', type=str)
arg_parser.add_argument('--file2', type=str)

args = arg_parser.parse_args()

tokenizer = NISTTokenizer()

with open(args.file1, 'r') as f1, open(args.fil2, 'r') as f2:
  for line1, line2 in zip(f1, f2):
    line1_toks = tokenizer.tokenize(line1)
    line2_toks = tokenizer.tokenize(line2)
    print(line1_toks)
    print(line2_toks)




