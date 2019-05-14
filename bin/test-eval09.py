import argparse

arg_parser = argparse.ArgumentParser(description='Output CoNLL-2009 file in eval09.pl format.')
arg_parser.add_argument('--input_file', type=str, help='CoNLL-2009 file to process')
arg_parser.add_argument('--output_file', type=str, help='File to write')
args = arg_parser.parse_args()

# input:
# 3	temperature	temperature	NOUN	NN	_	7	nsubjpass	Y	temperature.01	A2	A1	_	_
# want:
# 4	temperature	temperature	temperature	NN	NN	_	_	5	5	SBJ	SBJ	Y	temperature.01	A2	A1	_	_
with open(args.input_file) as f, open(args.output_file, 'w') as out_file:
  for line in f:
    line = line.strip()
    if line:
      split_line = line.split()
      id = split_line[0]
      word = split_line[1]
      gold_pos = split_line[3]
      dep_head = split_line[6]
      dep_label = split_line[7]
      rest = '\t'.join(split_line[8:])

      print("%s\t%s\t_\t_\t%s\t%s\t_\t_\t%s\t%s\t%s\t%s\t%s" % (id, word, gold_pos, gold_pos, dep_head, dep_head, dep_label, dep_label, rest), file=out_file)
    else:
      print("", file=out_file)
