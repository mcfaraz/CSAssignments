"""This module should filter and sort uk phone numbers from the text file provided. """

import sys
import re

if __name__ == "__main__":
    # Your code here
    inputFile = sys.argv[1]  # Input file argument

    # Open the input file and read all the lines
    with open(inputFile) as f:
        lines = f.readlines()

    phones = []
    for line in lines:
        # Strip the the newline character and all the white spaces
        line = line.strip().replace(' ', '')
        # Regex for finding numbers: +44 followed by only 10 digits
        p = re.compile(r'(\+44[0-9]{10})(?!\d)', re.M)
        matches = p.findall(line)
        for m in matches:
            m = m.replace("+44", '0')
            m = str(m[:5] + ' ' + m[5:])
            phones.append(m)
    phones.sort()
    for p in phones:
        print(p)
