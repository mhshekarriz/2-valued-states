# 2-valued-states
This package contains Python code for calculating all possible two-valued states of n-uniform (orthogonality) hypergraphs. Two-valued states are essential for quantum contextuality.

For technical definitions and more information, see my paper with Karl Svozil: https://pubs.aip.org/aip/jmp/article/63/3/032104/2845839/Noncontextual-coloring-of-orthogonality

The package contains 3 files:
1. hypergraph.py, which contains the class Hypergraph and required functions to generate all the two-valued states of an n-uniform hypergraph. It also has functions to analyse two-valued states.
2. states.py, which contains the old way of reading and writing text format.
3. modern_states.py, which can read not only old text format, but also reads from Excel files, and writes the two-valued states in an Excel file and a report txt file. It also generates a visualisation of the hypergraph.

To use the package, you need to have Python 3 installed and either use states.py or modern_states.py. The commends are as follows:
python states.py [input.txt] [optional: output file name]
or
python modern_states.py [input.txt or input.xlsx]

For the required formats of the input files, see the files in the folder Examples.
