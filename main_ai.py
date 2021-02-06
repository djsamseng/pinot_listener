import array
import numpy as np
import time

from multiprocessing import Process, Pipe, Queue
from types import SimpleNamespace

import ai
import audio
import sqlite_db_ai

# Get data from db
# Master transfers
# Each GPU holds state

NUM_INPUT_ONLY_NODES = 2048
s_tick = 0
DO_LOG_TIME = False

def audio_callback(in_data, weights, input_locs, output_locs, node_values, num_node_input_locs, num_node_output_locs):
    # len 2048 (NUM_INPUT_ONLY_NODES) each element is an int from [0, 255]
    for i in range(len(in_data)):
        # Set curr_values for the input nodes
        node_values[i] = in_data[i] / 256
    tick(weights=weights,
        input_locs=input_locs,
        output_locs=output_locs,
        node_values=node_values,
        num_node_input_locs=num_node_input_locs,
        num_node_output_locs=num_node_output_locs,
        max_input_node_id=len(in_data))

# TODO gather inputs from value_distributed instead of the node values
def tick(weights, input_locs, output_locs, node_values, num_node_input_locs, num_node_output_locs, max_input_node_id):
    global s_tick, DO_LOG_TIME
    begin_time = time.time()
    print("Start tick {0}".format(s_tick), flush=True)
    # First process the input only nodes
    assert(max_input_node_id == 2048)
    node_connectivity = 100
    resistance_threshold = 1
    node_threshold = 0.5
    learn_ratio = 0.0001

    for node_id in range(max_input_node_id):
        inputs_to_this_node = np.zeros((1, node_connectivity))
        input_weights_to_this_node = np.zeros((node_connectivity, 1))
        output_weights_from_this_node = np.ones((1, node_connectivity)) # TODO
        output_nodes_resistance = np.ones((1, node_connectivity)) * 0.1 # TODO
        after = ai.calc_single_node(inputs_to_this_node=inputs_to_this_node,
            input_weights_to_this_node=input_weights_to_this_node,
            this_nodes_orig_value=np.reshape(node_values[node_id], (1,1)),
            output_weights_from_this_node=output_weights_from_this_node,
            output_nodes_resistance=output_nodes_resistance,
            node_connectivity=node_connectivity,
            node_threshold=node_threshold,
            resistance_threshold=resistance_threshold,
            learn_ratio=learn_ratio)
        after = SimpleNamespace(**after)
        for i in range(num_node_output_locs[node_id]):
            dest = output_locs[node_id, i]
            # TODO do this for regular nodes
            # TODO finish refactor of node_values and copy as orig_values so that
            # inputs to the nodes are different that the nodes current_values
            node_values[dest] = after.outputs_from_this_node[0, i]


    for node_id in range(max_input_node_id, len(node_values)):
        if num_node_input_locs[node_id] == 0:
            pass
            #continue

        node_weights = weights[node_id]
        if np.sum(node_weights) < 0.00001:
            node_weights[:] = 1.0
            node_weights = node_weights / len(node_weights)
        input_array = node_values[input_locs[node_id]]
        input_array[input_array < 0] = 0
        if True:
            this_nodes_orig_value = np.reshape(node_values[node_id], (1, 1))

            inputs_to_this_node = np.reshape(input_array, (1, node_connectivity))
            input_weights_to_this_node = np.reshape(node_weights, (node_connectivity, 1))
            output_weights_from_this_node = np.reshape(node_weights, (1, node_connectivity))
            output_nodes_resistance = np.reshape(input_array, (1, node_connectivity))
            after = ai.calc_single_node(inputs_to_this_node=inputs_to_this_node,
                input_weights_to_this_node=input_weights_to_this_node,
                this_nodes_orig_value=this_nodes_orig_value,
                output_weights_from_this_node=output_weights_from_this_node,
                output_nodes_resistance=output_nodes_resistance,
                node_connectivity=node_connectivity,
                node_threshold=0.5,
                resistance_threshold=1,
                learn_ratio=0.001)
            after = SimpleNamespace(**after)
            # TODO take output weights into account
            # TODO update learned weights
            # TODO update node_values after all nodes have been calculated off of connection signals
            node_values[node_id] = after.this_nodes_after_value[0,0]
            #print(res)
        else:
            new_node_val = np.dot(input_array, node_weights) / num_node_input_locs[node_id]
            node_values[node_id] = node_values[node_id] * 0.9 + new_node_val * 0.1
            weight_change = (input_array - (256 / 2) ) / 256
            # print("Weight change: {0}".format(weight_change))
            node_weights = node_weights * 0.9 + weight_change * 0.1
            node_weights[node_weights > 1] = 1
            node_weights[node_weights < 0] = 0
            weights[node_id] = node_weights
            # 0.08s

    end_time = time.time()
    if DO_LOG_TIME:
        print("End tick {0} took:{1}".format(s_tick, end_time - begin_time), flush=True)
    s_tick += 1


def populate_input_array_for_node(input_array, all_nodes, node):
    idx = 0
    for node_id in node["input_connections"].keys():
        input_array[idx] = all_nodes[node_id]["value"]
        idx += 1

def get_nodes():
    # Get from sqlite3
    nodes = sqlite_db_ai.init_worker_sqlite(0, 1)
    num_nodes = len(nodes.keys())
    max_connections_per_node = 100
    # weights[0] is the input weights of node id 0
    # weights[0][3] is the input weight of node id 0's 3rd input connection into node id 0
    weights = np.zeros((num_nodes, max_connections_per_node))
    # Relies on weights being 0 for input_locs that don't exist
    input_locs = np.zeros((num_nodes, max_connections_per_node), dtype=int)
    output_locs = np.zeros((num_nodes, max_connections_per_node), dtype=int)
    node_values = np.zeros((num_nodes))
    num_node_input_locs = np.zeros((num_nodes), dtype=int)
    num_node_output_locs = np.zeros((num_nodes), dtype=int)

    i = 0
    j = 0
    output_locs_array = []
    for _ in range(num_nodes):
        output_locs_array.append([])
    for inputting_node_key, inputting_node in nodes.items():
        j = 0
        input_locs_for_node = []
        for outputting_node_key, connection in inputting_node["input_connections"].items():
            if j >= max_connections_per_node:
                break
            weights[i][j] = connection["weight"]
            input_locs_for_node.append(outputting_node_key)
            output_locs_array[outputting_node_key].append(inputting_node_key)
            j += 1
        input_locs[i][:len(input_locs_for_node)] = input_locs_for_node
        node_values[i] = inputting_node["value"]
        num_node_input_locs[i] = len(input_locs_for_node)
        i += 1
    for itr in range(num_nodes):
        num_node_output_locs[itr] = len(output_locs_array[itr])
        output_locs[itr, :len(output_locs_array[itr])] = output_locs_array[itr]

    print("Init weights:{0}".format(weights))
    print("Init values:{0}".format(node_values[NUM_INPUT_ONLY_NODES:]), flush=True)
    return {
        "input_locs": input_locs,
        "output_locs": output_locs,
        "nodes": nodes,
        "node_values": node_values,
        "num_node_input_locs": num_node_input_locs,
        "num_node_output_locs": num_node_output_locs,
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
    sqlite_db_ai.save_nodes(nodes)

def get_audio_output(node_values):
    output_values = node_values[-NUM_INPUT_ONLY_NODES:]
    output_values = np.round(output_values)
    output_values = output_values.astype(np.int16)
    output_values[output_values < 0] = 0
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

def audio_record_on_different_process(audio_child_conn, audio_record_queue, audio_play_queue):
    audio.record(audio_child_conn, audio_record_queue, audio_play_queue)

def data_manager_main(data_manager_child_conn):
    global DO_LOG_TIME
    node_data = get_nodes()
    input_locs = node_data["input_locs"]
    output_locs = node_data["output_locs"]
    nodes = node_data["nodes"]
    node_values = node_data["node_values"]
    num_node_input_locs = node_data["num_node_input_locs"]
    num_node_output_locs = node_data["num_node_output_locs"]
    weights = node_data["weights"]
    orig_weights = weights.copy()

    audio_parent_conn, audio_child_conn = Pipe()
    audio_record_queue = Queue()
    audio_play_queue = Queue()
    audio_process = Process(target=audio_record_on_different_process,
        args=(audio_child_conn, audio_record_queue, audio_play_queue))
    audio_process.start()

    do_save_recording = False
    recording_frames = []

    itr = 0
    while True:
        begin_time = time.time()
        if not audio_record_queue.empty():
            audio_data = audio_record_queue.get_nowait()
            print("Got audio_parent_conn: {0}".format(len(audio_data)))
            audio_callback(in_data=audio_data,
                weights=weights,
                input_locs=input_locs,
                output_locs=output_locs,
                node_values=node_values,
                num_node_input_locs=num_node_input_locs,
                num_node_output_locs=num_node_output_locs)
            if do_save_recording:
                recording_frames.append(audio_data)
            audio_output = get_audio_output(node_values)
            VALIDATE_AUDIO_OUTPUT = False
            if VALIDATE_AUDIO_OUTPUT:
                audio_output = []
                for i in range(len(audio_data)):
                    audio_output.append(audio_data[i])
            audio_output = array.array('B', audio_output).tobytes()
            audio_play_queue.put(audio_output)

        has_parent_data = data_manager_child_conn.poll(0.001)
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
        end_time = time.time()
        if DO_LOG_TIME:
            print("End loop {0} took:{1}".format(itr, end_time - begin_time), flush=True)

        itr += 1

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

