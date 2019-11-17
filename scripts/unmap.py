from __future__ import print_function
import sys

def main(index_file, score_file):
    mapping = {}
    with open(index_file) as f:
        for line in f:
            pieces = line.strip().split(None, 1)
            num = pieces[0]
            label = "" if len(pieces) == 1 else pieces[1]
            mapping[num] = label

    with open(score_file) as f:
        for line in f:
            line = line.strip()
            if line:
                id, score = line.split(": ")
                print("{} {}".format(mapping[id], score))
            else:
                print()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: `python unmap.py index_file score_file`")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

