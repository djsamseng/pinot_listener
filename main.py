import numpy as np

from multiprocessing import Process, Pipe

import audio
import sqlite_db

# Get data from db
# Master transfers
# Each GPU holds state

def audio_callback(in_data, nodes, weights):
    # len 2048 each element is an int from [0, 255]
    for i in range(len(in_data)):
        nodes[i]["value"] = in_data[i]
    tick(nodes, weights)

s_tick = 0
def tick(nodes, weights):
    global s_tick
    # print("Start tick {0}".format(s_tick), flush=True)
    for node_id, node in nodes.items():
        if len(node["input_connections"]) == 0:
            continue
        node_weights = weights[node_id]
        input_array = np.zeros_like(node_weights)
        populate_input_array_for_node(input_array, nodes, node)
        new_node_val = np.dot(input_array, node_weights) / len(node["input_connections"])
        node["value"] = node["value"] * 0.9 + new_node_val * 0.1
    # print("End tick {0}".format(s_tick), flush=True)
    s_tick += 1


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

def get_audio_output(nodes):
    output_nodes_keys = nodes.keys()
    output_values = np.zeros((2048))
    max_node_key = max(output_nodes_keys)
    for i in range(2048):
        node_id = max_node_key - 2048 + i
        output_values[i] = round(nodes[node_id]["value"])
    return output_values

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

def data_manager_main(data_manager_child_conn):
    node_data = get_nodes()
    nodes = node_data["nodes"]
    weights = node_data["weights"]

    audio_parent_conn, audio_child_conn = Pipe()
    audio_process = Process(target=audio_record_on_different_process, args=(audio_child_conn,))
    audio_process.start()

    while True:
        has_audio_data = audio_parent_conn.poll(0.01)
        if has_audio_data:
            audio_data = audio_parent_conn.recv()
            print("Got audio_parent_conn: {0}".format(len(audio_data)))
            audio_callback(audio_data, nodes, weights)
            audio_output = get_audio_output(nodes)
            # FIXME
            audio_output = audio_data
            audio_parent_conn.send({
                "key": "play_audio",
                "data": bytes(audio_output)
            })

        has_parent_data = data_manager_child_conn.poll(0.01)
        if has_parent_data:
            parent_data = data_manager_child_conn.recv()
            # print("Got data_manager_child_conn data: {0}".format(parent_data))
            if parent_data and parent_data["key"] == "exit":
                break

    audio_parent_conn.send({
        "key": "exit"
    })
    print_node_values(nodes)
    audio_process.join()
    print("Closed data_manager_main", flush=True)

def main():
    # Command processor process

    # Data manager process
    data_manager_parent_conn, data_manager_child_conn = Pipe()
    data_manager = Process(target=data_manager_main, args=(data_manager_child_conn,))
    data_manager.start()

    command = None
    while command is not "e":
        command = input("Enter command: ")

    data_manager_parent_conn.send({
        "key": "exit"
    })

    print("Waiting to join processes")
    data_manager.join()
    print("Joined processes")


if __name__ == "__main__":
    main()

