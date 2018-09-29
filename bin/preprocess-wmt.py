import argparse
import time
from nltk.tokenize.nist import NISTTokenizer

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--source_file', type=str)
arg_parser.add_argument('--target_file', type=str)

args = arg_parser.parse_args()

tokenizer = NISTTokenizer()

source_max_len = 0
target_max_len = 0
target_2x_source = 0
line_count = 0
start_time = time.time()
with open(args.source_file, 'r') as source, open(args.target_file, 'r') as target:
  for source_line, target_line in zip(source, target):
    source_toks = tokenizer.tokenize(source_line)
    target_toks = tokenizer.tokenize(target_line)
    source_len = len(source_toks)
    target_len = len(target_toks)
    if source_len > source_max_len:
      source_max_len = source_len
    if target_len > target_max_len:
      target_max_len = target_len

    if target_len > 2 * source_len:
      target_2x_source += 1
    line_count += 1

    if line_count % 10000 == 0:
      print("Processed %d lines" % line_count)

print("Processed %d lines in %d seconds" % (line_count, time.time() - start_time))
print("max source len: %d" % source_max_len)
print("max target len: %d" % target_max_len)
print("target > 2 * source: %d" % target_2x_source)






