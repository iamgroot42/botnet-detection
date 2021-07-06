from background import prepare_background_
import os
import argparse
from tqdm import tqdm
import networkx as nx
import deepdish as dd


def get_file_names(file_list, basepath):
    names = []
    with open(file_list, 'r') as f:
        for line in f:
            name = line.rstrip('\n')
            names.append(os.path.join(basepath, name))
    return names


def make_networkx_from_custom(num_nodes, edge_index):
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(
        zip(edge_index[0],edge_index[1]))
    return g.to_undirected()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type=str,
                        default="/p/adversarialml/as9rw/datasets/raw_botnet")
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()

    # Load all names
    names = get_file_names(args.filepath, args.basepath)

    for name in tqdm(names):

        # Check if file exists
        if not os.path.exists(name):
            continue

        # Generate graph
        graph_name = name[:-5] + '.hdf5'
        
        # Get graph in networkX form
        edge_index, num_nodes, num_edges = prepare_background_(name, None, None)
        graph = make_networkx_from_custom(num_nodes, edge_index)

        # Save for later
        dd.io.save(os.path.join(args.basepath, graph_name), graph)

        # Remove temp file
        # os.remove(name)
