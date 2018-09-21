from collections import defaultdict
import argparse

arg_parser = argparse.ArgumentParser(description='Compute transition probabilities between SRL tags')
arg_parser.add_argument('--in_file_name', type=str, help='File to process')

args = arg_parser.parse_args()

with open(args.in_file_name, 'r') as in_f:
    # keep track of label_a / label_b counts here
    label_counts = defaultdict(dict)
    current_sentence = []
    current_domain = None
    for line_num, line in enumerate(in_f):
        line = line.strip()
        # blank line means end of sentence
        if not line:
            sentence_frames = []
            # grab only columns corresponding to frames
            for token_parts in current_sentence:
                parts = token_parts.split('\t')
                # first 13 columns are not frames, last column is coref for some reason
                if len(parts) > 14:
                    frames = parts[14:-1]
                    sentence_frames.append(frames)
            # seperate out into list per frame sequence
            frame_sequences = zip(*sentence_frames)
            for frame in frame_sequences:
                if frame:
                    for i in range(len(frame)-1):
                        # keep track of counts for each label pair transition
                        label_a = frame[i]
                        label_b = frame[i+1]
                        label_a_count_map = label_counts[label_a]
                        if label_b in label_a_count_map:
                            label_a_count_map[label_b] += 1
                        else:
                            label_a_count_map[label_b] = 1
            current_sentence = []
        else:
            current_sentence.append(line)

for label_a, transitions in label_counts.iteritems():
    total_count = float(sum([count for label, count in transitions.iteritems()]))
    for label_b, count in transitions.iteritems():
        print('%s\t%s\t%g' % (label_a, label_b, (count/total_count)))
