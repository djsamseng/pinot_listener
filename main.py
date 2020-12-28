import numpy as np
import time

from multiprocessing import Process, Pipe, Queue

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
    do_log = False
    begin_time = time.time()
    # print("Start tick {0}".format(s_tick), flush=True)
    for node_id, node in nodes.items():
        if len(node["input_connections"]) == 0:
            continue
        node_weights = weights[node_id]
        input_array = np.zeros_like(node_weights)
        # 0.03s
        populate_input_array_for_node(input_array, nodes, node)
        # 0.08s
        new_node_val = np.dot(input_array, node_weights) / len(node["input_connections"])
        # 0.10s
        node["value"] = node["value"] * 0.9 + new_node_val * 0.1
        # 0.11s
        weight_change = (input_array - (256 / 2) )/ 256
        # print("Weight change: {0}".format(weight_change))
        # 0.14s
        node_weights = node_weights * 0.9 + weight_change * 0.1
        # 0.16s
        node_weights = np.clip(node_weights, -1, 1)
        # 0.27-0.34s
        weights[node_id] = node_weights
        # 0.25-0.36s

    end_time = time.time()
    if do_log:
        print("End tick {0} took:{1}".format(s_tick, end_time - begin_time), flush=True)
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
    weights = np.zeros((num_nodes, max_connections_per_node))
    i = 0
    j = 0
    for inputting_node_key, inputting_node in nodes.items():
        j = 0
        for outputting_node_key, connection in inputting_node["input_connections"].items():
            weights[i][j] = connection["weight"]
            j += 1
        i += 1
    print("Init weights{0}".format(weights))
    return {
        "nodes": nodes,
        "weights": weights
    }

def save_nodes(nodes, weights):
    i = 0
    j = 0
    for inputting_node_key, inputting_node in nodes.items():
        j = 0
        for outputting_node_key, connection in inputting_node["input_connections"].items():
            connection["weight"] = weights[i][j]
            j += 1
        i += 1
    sqlite_db.save_nodes(nodes)

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

def audio_record_on_different_process(audio_child_conn, audio_record_queue):
    audio.record(audio_child_conn, audio_record_queue)

def data_manager_main(data_manager_child_conn):
    node_data = get_nodes()
    nodes = node_data["nodes"]
    weights = node_data["weights"]
    orig_weights = weights.copy()

    audio_parent_conn, audio_child_conn = Pipe()
    audio_record_queue = Queue()
    audio_process = Process(target=audio_record_on_different_process,
        args=(audio_child_conn, audio_record_queue))
    audio_process.start()

    do_save_recording = False
    recording_frames = []

    while True:
        if not audio_record_queue.empty():
            audio_data = audio_record_queue.get_nowait()
            # print("Got audio_parent_conn: {0}".format(len(audio_data)))
            audio_callback(audio_data, nodes, weights)
            if do_save_recording:
                recording_frames.append(audio_data)
            audio_output = get_audio_output(nodes)
            # FIXME - need to do this on another process to not block audio recording speed
            # audio_output = audio_data
            if False:
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
            if parent_data and parent_data["key"] == "print_weights":
                print("Weights: {0}".format(weights))
            if parent_data and parent_data["key"] == "save":
                print("Saving weights:{0}".format(weights))
                save_nodes(nodes, weights)
            if parent_data and parent_data["key"] == "begin_save_recording":
                print("Saving recording")
                do_save_recording = True
                recording_frames = []
            if parent_data and parent_data["key"] == "end_save_recording" and do_save_recording:
                filename = "audio_recording.wav"
                audio.save_recording(recording_frames, filename)
                print("Saved recording to:{0} with frames:{1}".format(filename, len(recording_frames)))
                do_save_recording = False

    audio_parent_conn.send({
        "key": "exit"
    })
    audio_parent_conn.close()
    print_node_values(nodes)
    print("New weights: {0}".format(weights))
    print("Weights changed: {0}".format(not (weights == orig_weights).all()))
    extra_record_data = []
    while not audio_record_queue.empty():
        extra_record_data.append(audio_record_queue.get())
    print("Joining audio. Extra record data: {0}".format(len(extra_record_data)), flush=True)
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
        if command == "pw":
            data_manager_parent_conn.send({
                "key": "print_weights"
            })
        if command == "save":
            data_manager_parent_conn.send({
                "key": "save"
            })
        if command == "begin_save_recording":
            data_manager_parent_conn.send({
                "key": "begin_save_recording"
            })
        if command == "end_save_recording":
            data_manager_parent_conn.send({
                "key": "end_save_recording"
            })

    data_manager_parent_conn.send({
        "key": "exit"
    })

    print("Waiting to join processes")
    data_manager.join()
    print("Joined processes")


if __name__ == "__main__":
    main()

