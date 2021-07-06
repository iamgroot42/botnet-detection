from download_pcap import decompress
# from background import prepare_background_
import os
import argparse
from tqdm import tqdm
import networkx as nx
import deepdish as dd


def get_file_names(download_list_path, basepath):
    names = []
    with open(download_list_path, 'r') as f:
        for line in f:
            name = line.rstrip('\n').split('/')[-1]
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
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--basepath', type=str,
                        default="/p/adversarialml/as9rw/datasets/raw_botnet")
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()

    # Load all names
    names = get_file_names(args.filepath, args.basepath)

    # Get relevant split of data (for parallel processing)
    shard_size = len(names) // args.n_splits
    use_names = names[args.id * shard_size : (args.id + 1) * shard_size]

    # Only generating PCAP files for now
    for name in tqdm(use_names):
        # Decompress file
        name = decompress(name, basepath=args.basepath)
