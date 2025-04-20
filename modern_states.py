import sys
import pandas as pd
from typing import List
from hypergraph import Hypergraph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def parse_hypergraph(filename):
    if filename.endswith(".txt"):
        with open(filename, 'r') as f:
            lines = f.readlines()

        name = lines[0].strip()
        num_atoms = int(lines[1].split()[0])
        num_blocks = int(lines[2].split()[0])
        num_subsets = int(lines[3].split()[0])

        blocks = []
        for i in range(4, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            if ":" in line:
                break
            parts = line.split()
            try:
                block = list(map(int, parts[1:]))
                blocks.append(block)
            except ValueError:
                break
            if len(blocks) == num_blocks:
                break
        return name, num_atoms, blocks

    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename, header=None)
        hyperedges = [row.dropna().astype(int).tolist() for index, row in df.iterrows()]
        num_atoms = 0
        if hyperedges:
            for he in hyperedges:
                num_atoms = max(num_atoms, max(he) if he else 0)
        return filename.split('.')[0], num_atoms, hyperedges
    else:
        raise ValueError(f"Unsupported file format: {filename}. Please use .txt or .xlsx.")

def is_unital_check(num_vertices: int, states: List[List[int]]) -> List[int]:
    non_unital_vertices = []
    for i in range(num_vertices):
        if not any(state[i] == 1 for state in states):
            non_unital_vertices.append(i + 1)
    return non_unital_vertices

def write_reports(filename, hypergraph_name, num_vertices, hyperedges, states, separable, non_unital, tifs_pairs, tits_pairs, not_pairs, and_triples):
    with open(filename, 'w') as f:
        f.write(f"Hypergraph Name: {hypergraph_name}\n")
        f.write(f"Number of vertices: {num_vertices}\n")
        f.write(f"Hyperedges: {hyperedges}\n")
        f.write(f"Number of valid 2-valued states: {len(states)}\n")
        f.write("Separable: " + ("yes" if separable else "no") + "\n")
        if non_unital:
            f.write("Unital: no for vertices " + ' '.join(map(str, non_unital)) + "\n")
        else:
            f.write("Unital: yes\n")
        if tifs_pairs:
            f.write("TIFS pairs (u -> not v) where u < v: " + ', '.join(map(str, tifs_pairs)) + "\n")
        else:
            f.write("No TIFS pairs found.\n")
        if tits_pairs:
            f.write("TITS pairs (u -> v) where u < v: " + ', '.join(map(str, tits_pairs)) + "\n")
        else:
            f.write("No TITS pairs found.\n")
        if not_pairs:
            f.write("NOT pairs (u <-> not v): " + ', '.join(map(str, not_pairs)) + "\n")
        else:
            f.write("No NOT pairs found.\n")
        if and_triples:
            f.write("AND triples (u AND v = w): " + ', '.join(map(str, and_triples)) + "\n")
        else:
            f.write("No AND triples found.\n")

def visualize_hypergraph_lines_through_vertices(num_vertices: int, hyperedges: List[List[int]], filename_prefix: str):
    """Visualizes hyperedges as lines connecting their vertices."""
    G = nx.Graph()
    for v in range(1, num_vertices + 1):
        G.add_node(v)

    pos = nx.spring_layout(G, seed=42)  # Layout for vertex positions

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)

    colors = [plt.cm.viridis(i/len(hyperedges)) for i in range(len(hyperedges))]
    random.shuffle(colors)

    for i, he in enumerate(hyperedges):
        if len(he) >= 2:
            for j in range(len(he) - 1):
                u, v = he[j], he[j + 1]
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                plt.plot(x, y, color=colors[i], alpha=0.7, linewidth=2)
        elif len(he) == 1:
            u = he[0]
            plt.scatter(pos[u][0], pos[u][1], color=colors[i], alpha=0.7, s=100)

    plt.title("Hypergraph Visualization (Lines Through Vertices)")
    plt.axis('off')
    plt.savefig(f"{filename_prefix}_hypergraph_lines_through.png")
    plt.show()

def main(input_file):
    hypergraph_name, num_atoms, hyperedges = parse_hypergraph(input_file)
    hg = Hypergraph(num_vertices=num_atoms, hyperedges=hyperedges)
    states = hg.generate_valid_states()
    separable = hg.is_separating(states)
    non_unital = is_unital_check(num_atoms, states)
    tifs_pairs = hg.is_tifs(states)
    tits_pairs = hg.is_tits(states)
    not_pairs = hg.is_not(states)
    and_triples = hg.is_and(states)

    output_states_file = f"{hypergraph_name}_states.xlsx"
    output_reports_file = f"{hypergraph_name}_reports.txt"
    output_prefix = hypergraph_name

    header = [f"v{i+1}" for i in range(num_atoms)]
    states_df = pd.DataFrame(states, columns=header)
    states_df.insert(0, "State No.", range(1, len(states) + 1))
    states_df.to_excel(output_states_file, index=False)
    print(f"Two-valued states saved to: {output_states_file}")

    write_reports(output_reports_file, hypergraph_name, num_atoms, hyperedges, states, separable, non_unital, tifs_pairs, tits_pairs, not_pairs, and_triples)
    print(f"Hypergraph reports saved to: {output_reports_file}")

    # Call the line-through-vertices visualization function
    visualize_hypergraph_lines_through_vertices(num_atoms, hyperedges, output_prefix)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python modern_states.py <input_file.txt or input_file.xlsx>")
        sys.exit(1)
    input_file = sys.argv[1]
    main(input_file)