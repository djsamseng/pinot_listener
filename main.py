import numpy as np

import audio
import sqlite_db

# Get data from db
# Master transfers
# Each GPU holds state

# Copy circular results - copy from right to left
def get_a_from_mem(mem):
    a = mem[0][:]
    a[4] = mem[1][2]
    return a

def get_b_from_mem(mem):
    b = mem[1][:]
    return b

def demo_slice_ref():
    mem = np.zeros((3, 5))
    a = get_a_from_mem(mem)
    b = get_b_from_mem(mem)
    a[:] = 1
    b[:] = 2
    print(mem)
    print(a)
    print(b)
    print("------")
    mem[1][2] = 3
    print(mem)
    print(a)
    print(b)
    print("------")
    a = get_a_from_mem(mem)
    b = get_b_from_mem(mem)
    print(mem)
    print(a)
    print(b)

mem = np.zeros((4, 2048))
def audio_callback(in_data):
    # len 2048 each element is an int from [0, 255]
    print("GOT AUDIO CALLBACK")


def get_input_array_for_node(all_nodes, node):
    input_array = np.zeros((len(node["input_connections"].keys())))
    idx = 0
    for _, val in node["input_connections"].items():
        input_array[idx] = all_nodes[val]["value"]
        idx += 1
    return input_array

def get_nodes():
    # Get from sqlite3
    nodes = sqlite_db.init_worker_sqlite(0, 1)
    num_nodes = len(nodes.keys())
    max_connections_per_node = 100
    # mem[0] is all the output signals of node 0 to it's output_connections
    # mem[0][3] is the input weights of node id 0
    mem = np.zeros((num_nodes, max_connections_per_node))
    node_layers = []
    for layer in mem:
        node_layers.append(layer)
    # Go backwards - updates happen left to right but we need to pull in new inputs right to left
    node_mapping = {}

    for i in range(num_nodes, -1, -1):
        pass

    return {
        "node_mem": mem,
        "node_layers": node_layers
    }

def main():
    demo_slice_ref()
    # audio.record(audio_callback)


if __name__ == "__main__":
    main()

