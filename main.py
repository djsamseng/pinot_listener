import numpy as np
import time

from multiprocessing import Process, Pipe, Queue

import audio
import sqlite_db

# Get data from db
# Master transfers
# Each GPU holds state

NUM_INPUT_ONLY_NODES = 2048

def audio_callback(in_data, weights, input_locs, node_values, num_node_input_locs):
    # len 2048 (NUM_INPUT_ONLY_NODES) each element is an int from [0, 255]
    for i in range(len(in_data)):
        node_values[i] = in_data[i]
    tick(weights, input_locs, node_values, num_node_input_locs)

s_tick = 0
def tick(weights, input_locs, node_values, num_node_input_locs):
    global s_tick
    do_log = False
    begin_time = time.time()
    # print("Start tick {0}".format(s_tick), flush=True)
    for node_id in range(len(node_values)):
        if num_node_input_locs[node_id] == 0:
            continue
        node_weights = weights[node_id]
        input_array = node_values[input_locs[node_id]]
        new_node_val = np.dot(input_array, node_weights) / num_node_input_locs[node_id]
        node_values[node_id] = node_values[node_id] * 0.9 + new_node_val * 0.1
        weight_change = (input_array - (256 / 2) ) / 256
        # print("Weight change: {0}".format(weight_change))
        node_weights = node_weights * 0.9 + weight_change * 0.1
        node_weights[node_weights > 1] = 1
        node_weights[node_weights < -1] = -1
        weights[node_id] = node_weights
        # 0.08s

    end_time = time.time()
    if do_log:
        print("End tick {0} took:{1}".format(s_tick, end_time - begin_time), flush=True)
    s_tick += 1


def populate_input_array_for_node(input_array, all_nodes, node):
    idx = 0
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
    # Relies on weights being 0 for input_locs that don't exist
    input_locs = np.zeros((num_nodes, max_connections_per_node), dtype=int)
    node_values = np.zeros((num_nodes))
    num_node_input_locs = np.zeros((num_nodes))

    i = 0
    j = 0
    for inputting_node_key, inputting_node in nodes.items():
        j = 0
        input_locs_for_node = []
        for outputting_node_key, connection in inputting_node["input_connections"].items():
            weights[i][j] = connection["weight"]
            input_locs_for_node.append(outputting_node_key)
            j += 1
        input_locs[i][:len(input_locs_for_node)] = input_locs_for_node
        node_values[i] = inputting_node["value"]
        num_node_input_locs[i] = len(input_locs_for_node)
        i += 1
    print("Init weights:{0}".format(weights))
    print("Init values:{0}".format(node_values[NUM_INPUT_ONLY_NODES:]), flush=True)
    return {
        "input_locs": input_locs,
        "nodes": nodes,
        "node_values": node_values,
        "num_node_input_locs": num_node_input_locs,
        "weights": weights
    }

def save_nodes(nodes, weights, node_values):
    i = 0
    j = 0
    for inputting_node_key, inputting_node in nodes.items():
        j = 0
        inputting_node["value"] = node_values[inputting_node_key]
        for outputting_node_key, connection in inputting_node["input_connections"].items():
            connection["weight"] = weights[i][j]
            j += 1
        i += 1
    sqlite_db.save_nodes(nodes)

def get_audio_output(node_values):
    output_values = node_values[-NUM_INPUT_ONLY_NODES:]
    output_values = np.round(output_values)
    return output_values

def print_node_values(node_values):
    idxes_positive = []
    idxes_negative = []
    for i in range(len(node_values)):
        if node_values[i] > 0:
            idxes_positive.append(i)
        else:
            idxes_negative.append(i)
    print("Node values > 0 idxes: {0}".format(idxes_positive[:5]))
    print("Node values 0 idxes: {0}".format(idxes_negative[:5]))
    print("Number of node values greater than 0: {0}".format(len(node_values[node_values > 0])))

def audio_record_on_different_process(audio_child_conn, audio_record_queue):
    audio.record(audio_child_conn, audio_record_queue)

def data_manager_main(data_manager_child_conn):
    node_data = get_nodes()
    input_locs = node_data["input_locs"]
    nodes = node_data["nodes"]
    node_values = node_data["node_values"]
    num_node_input_locs = node_data["num_node_input_locs"]
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
            audio_callback(audio_data, weights, input_locs, node_values, num_node_input_locs)
            if do_save_recording:
                recording_frames.append(audio_data)
            audio_output = get_audio_output(node_values)
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
            if parent_data and parent_data["key"] == "print_values":
                print("Values: {0}".format(node_values))
            if parent_data and parent_data["key"] == "save":
                print("Saving weights:{0}".format(weights))
                print("Saving values:{0}".format(node_values[NUM_INPUT_ONLY_NODES:]))
                save_nodes(nodes, weights, node_values)
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
    print_node_values(node_values)
    print("New weights: {0}".format(weights))
    print("Weights changed: {0}".format(not (weights == orig_weights).all()))
    extra_record_data = []
    # If we don't pop everything off the queue then joining the process will be a deadlock
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
        if command == "pv":
            data_manager_parent_conn.send({
                "key": "print_values"
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

