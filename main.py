import numpy as np

from multiprocessing import Process, Pipe


import audio
import sqlite_db

# Get data from db
# Master transfers
# Each GPU holds state


GLOBALS = {
    "nodes": {},
    "exit": False
}
def audio_callback(in_data):
    # len 2048 each element is an int from [0, 255]
    global GLOBALS
    for i in range(len(in_data)):
        GLOBALS["nodes"][i]["value"] = in_data[i]
    tick()

def tick():
    global GLOBALS
    for node_id, node in GLOBALS["nodes"].items():
        if len(node["input_connections"]) == 0:
            continue
        node_weights = GLOBALS["weights"][node_id]
        input_array = np.zeros_like(node_weights)
        populate_input_array_for_node(input_array, GLOBALS["nodes"], node)
        new_node_val = np.dot(input_array, node_weights)
        node["value"] = new_node_val


def populate_input_array_for_node(input_array, all_nodes, node):
    idx = 0
    # For some reason having this loop iterate causes stream.stop_stream() to hang
    for node_id in node["input_connections"].keys():
        input_array[idx] = all_nodes[node_id]["value"]
        idx += 1

def get_nodes():
    # Get from sqlite3
    nodes = sqlite_db.init_worker_sqlite(0, 1)
    num_nodes = len(nodes.keys())
    max_connections_per_node = 100
    # weights[0] is the input weights of node id 0
    # weights[0][3] is the input weight of node id 0's 3rd input connection into node id 0
    weights = np.ones((num_nodes, max_connections_per_node))
    return {
        "nodes": nodes,
        "weights": weights
    }

def print_node_values(nodes):
    print_ar = np.zeros((len(nodes)))
    idx = 0
    for _, node in nodes.items():
        print_ar[idx] = node["value"]
        idx += 1
    idxes_positive = []
    idxes_negative = []
    for i in range(len(print_ar)):
        if print_ar[i] > 0:
            idxes_positive.append(i)
        else:
            idxes_negative.append(i)
    print("Node values > 0 idxes: {0}".format(idxes_positive[:5]))
    print("Node values 0 idxes: {0}".format(idxes_negative[:5]))
    print("Number of node values greater than 0: {0}".format(len(print_ar[print_ar > 0])))

def audio_record_on_different_process(audio_child_conn):
    audio.record(audio_child_conn)

def main():
    global GLOBALS
    node_data = get_nodes()
    GLOBALS["nodes"] = node_data["nodes"]
    GLOBALS["weights"] = node_data["weights"]

    audio_parent_conn, audio_child_conn = Pipe()
    audio_process = Process(target=audio_record_on_different_process, args=(audio_child_conn,))
    audio_process.start()

    while True:
        has_audio_data = audio_parent_conn.poll(0.1)
        if has_audio_data:
            audio_data = audio_parent_conn.recv()
            print("Got audio data {0}".format(len(audio_data)))
            audio_callback(audio_data)
            break

    audio_parent_conn.send({
        "key": "exit"
    })
    print_node_values(GLOBALS["nodes"])

    print("Waiting to join processes")
    audio_process.join()
    print("Joined processes")


if __name__ == "__main__":
    main()

