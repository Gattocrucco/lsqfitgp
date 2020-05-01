"""Search for rst files in the current directory and run the python code in
them. Or, run the files specified on the command line."""

import glob
import re
import sys
import numpy as np

sys.path.insert(0, '..')

pattern = re.compile(r'::\n\s*?\n(( {4,}.*\n)+)\s*?\n')

def runcode(file):
    with open(file, 'r') as stream:
        text = stream.read()
    globals = {}
    np.random.seed(0)
    for match in pattern.finditer(text):
        codeblock = match.group(1)
        print(codeblock)
        code = '\n'.join(line[4:] for line in codeblock.split('\n'))
        exec(code, globals, globals)

files = sys.argv[1:]
if not files:
    files = glob.glob('*.rst')
    files.sort()
for file in files:
    print('running {}...'.format(file))
    runcode(file)
