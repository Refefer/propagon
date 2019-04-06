import sys

def main(fname, delim=None):
    with open(fname + '.remap', 'w') as mout, open(fname) as f:
        m = {}
        for line in f:
            w, l = line.strip().split(delim)
            if w not in m:
                m[w] = len(m)

            if l not in m:
                m[l] = len(m)

            mout.write('{} {}\n'.format(m[w], m[l]))

    with open(fname + '.ids', 'w') as iout:
        for name, idx in sorted(m.items()):
            iout.write('{} {}\n'.format(idx, name))

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print("Usage: `python remap.py comparison_file <delim>`")
        sys.exit(1)

    delim = None if len(sys.argv) == 2 else sys.argv[2]
        
    main(sys.argv[1], delim)

