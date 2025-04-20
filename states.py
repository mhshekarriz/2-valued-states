import sys
import itertools
from collections import defaultdict
from hypergraph import Hypergraph  # Import the Hypergraph class

def parse_hypergraph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    name = lines[0].strip()
    num_atoms = int(lines[1].split()[0])
    num_blocks = int(lines[2].split()[0])
    num_subsets = int(lines[3].split()[0])

    blocks = []
    for line in lines[4:4 + num_blocks]:
        parts = list(map(int, line.strip().split()))
        blocks.append(parts[1:])

    return name, num_atoms, blocks, lines[:4 + num_blocks]

def is_unital_check(num_vertices: int, states: list[list[int]]) -> list[int]:
    """Check if the hypergraph is unital and return a list of non-unital vertices."""
    non_unital_vertices = []
    for i in range(num_vertices):
        if not any(state[i] == 1 for state in states):
            non_unital_vertices.append(i + 1)  # Store 1-based vertex index
    return non_unital_vertices

def write_output(filename, header_lines, states, separable, non_unital):
    with open(filename, 'w') as f:
        for line in header_lines:
            f.write(line)
        f.write("\n")
        f.write(f"{len(states)} 2-valued evaluations of atoms:\n")
        for state in states:
            f.write(' '.join(map(str, state)) + '\n')
        f.write("\nset of 2-valued evaluations of atoms:\n")
        f.write("nonempty: " + ("yes" if states else "no") + "\n")
        f.write("Separable: " + ("yes" if separable else "no") + "\n")
        if non_unital:
            f.write("unital: no for atoms " + ' '.join(map(str, non_unital)) + "\n")
        else:
            f.write("unital: yes\n")


def main(input_file, output_file=None):
    output_file = output_file or input_file
    name, num_atoms, blocks, header = parse_hypergraph(input_file)
    atoms= [i for i in range(1, num_atoms + 1)]
    hg = Hypergraph(num_vertices=num_atoms, hyperedges=blocks)
    states = hg.generate_valid_states()
    separable = hg.is_separating(states)
    non_unital = is_unital_check(num_atoms, states) # Use the new function
    hg.analyze() # This will print to the console
    write_output(output_file, header, states, separable, non_unital)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file name")
    parser.add_argument("output_file", nargs='?', default=None, help="Optional output file name")
    args = parser.parse_args()
    main(args.input_file, args.output_file)