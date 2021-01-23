import array
import json
import numpy as np
import os
import time
from enum import Enum

from functools import partial
from glob import glob
from multiprocessing import Process, Pipe, Queue
from types import SimpleNamespace

import audio2

'''
The idea is that each at each tick the AI decides which inputs
to replay onto the next tick's inputs
Ex:
"Hi there" is heard
H i  t h e r e
So we recognize the "H" and save it since it's relevant and the next
tick we form "Hi" then "Hi", "Hi ", "Hi t", ..., "Hi there"
and the AI keeps saving up these activations until it no longer needs it
because it formed a higher level concept and the lower level concepts
are less activated unless it decides it needs them again
(because maybe it was the wrong higher level concept)
'''

class KEYS(Enum):
    EXIT = 1
    PLAY_RECORDING = 2
    START_RECORDING = 3
    STOP_RECORDING = 4

def audio_record_on_different_process(audio_child_conn, audio_record_queue, audio_play_queue):
    audio2.record(audio_child_conn, audio_record_queue, audio_play_queue)


def get_data_manager_conn():
    audio_parent_conn, audio_child_conn = Pipe()
    audio_record_queue = Queue()
    audio_play_queue = Queue()
    audio_process = Process(target=audio_record_on_different_process,
        args=(audio_child_conn, audio_record_queue, audio_play_queue))
    audio_process.start()

    return {
        "audio_parent_conn": audio_parent_conn,
        "audio_child_conn": audio_child_conn,
        "audio_record_queue": audio_record_queue,
        "audio_play_queue": audio_play_queue,
        "audio_process": audio_process,
    }

def audio_queue_get_all(queue):
    data = []
    while not queue.empty():
        audio_data = queue.get_nowait()
        for i in range(len(audio_data)):
            data.append(audio_data[i])
    return data

def gather_until_exit(queue):
    audio_queue_get_all(queue)
    command = None
    while command is not "e":
        command = input("Press e to stop recording")
    return audio_queue_get_all(queue)

def save_audio_data(filename, audio_arr):
    filename = filename + ".json"
    with open(filename, "r") as f:
        f.read()
    with open(filename, "w") as f:
        f.write(json.dumps(audio_arr))

def read_audio_data(filename):
    with open(filename, "r") as f:
        play_data = json.loads(f.read())
    return play_data

def read_concepts():
    ret = {}
    for filename in glob("./*.json"):
        end_index = filename.find(".json")
        concept_name = filename[2:end_index]
        ret[concept_name] = read_audio_data(filename)
    return SimpleNamespace(**ret)

def play_audio_data(audio_play_queue, play_arr):
    audio_output = array.array('B', play_arr).tobytes()
    audio_play_queue.put_nowait(audio_output)

def data_manager_api():
    data_manager_cons = get_data_manager_conn()
    aud = SimpleNamespace(**data_manager_cons)
    return aud

def data_manager_main(data_manager_child_conn):
    audio_parent_conn, audio_child_conn = Pipe()
    audio_record_queue = Queue()
    audio_play_queue = Queue()
    audio_process = Process(target=audio_record_on_different_process,
        args=(audio_child_conn, audio_record_queue, audio_play_queue))
    audio_process.start()

    itr = 0
    record_data = []
    record_name = None
    while True:
        if not audio_record_queue.empty():
            audio_data = audio_record_queue.get_nowait()
            print("Got audio_parent_conn: {0}".format(len(audio_data)))
            for i in range(len(audio_data)):
                record_data.append(audio_data[i])
            # audio_output = array.array('B', audio_output).tobytes()
            # audio_play_queue.put(audio_output)

        has_parent_data = data_manager_child_conn.poll(0.001)
        if has_parent_data:
            parent_data = data_manager_child_conn.recv()
            print("Got data_manager_child_conn data: {0}".format(parent_data))
            if parent_data["key"] == KEYS.EXIT:
                print("GOT exit", flush=True)
                break
            if parent_data["key"] == KEYS.PLAY_RECORDING:
                filename = parent_data["name"] + ".json"
                with open(filename, "r") as f:
                    play_data = json.loads(f.read())
                    print("Got plat", play_data)
                    audio_output = array.array('B', play_data).tobytes()
                    audio_play_queue.put_nowait(audio_output)
            if parent_data["key"] == KEYS.START_RECORDING:
                while not audio_record_queue.empty():
                    audio_record_queue.get_nowait()
                record_data = []
                record_name = parent_data["name"]
            if parent_data["key"] == KEYS.STOP_RECORDING:
                if not record_name:
                    print("No recording to stop", flush=True)
                    continue
                filename = record_name + ".json"
                with open(filename, "r") as f:
                    f.read()
                with open(filename, "w") as f:
                    print(record_data, flush=True)
                    f.write(json.dumps(record_data))
                record_name = None

        itr += 1

    print("Send exit", flush=True)
    audio_parent_conn.send({
        "key": "exit"
    })
    print("Close audio_parent", flush=True)
    audio_parent_conn.close()

    extra_record_data = []
    # If we don't pop everything off the queue then joining the process will be a deadlock
    while not audio_record_queue.empty():
        extra_record_data.append(audio_record_queue.get())
    print("Joining audio. Extra record data: {0}".format(len(extra_record_data)), flush=True)
    audio_process.join()
    print("Closed data_manager_main", flush=True)

def main():
    # Data manager process
    data_manager_parent_conn, data_manager_child_conn = Pipe()
    data_manager = Process(target=data_manager_main, args=(data_manager_child_conn,))
    data_manager.start()

    command = None
    while command is not "e":
        command = input("Enter command: ")
        if command.startswith("r "):
            data_manager_parent_conn.send({
                "key": KEYS.START_RECORDING,
                "name": command[2:]
            })
        if command == "s":
            data_manager_parent_conn.send({
                "key": KEYS.STOP_RECORDING,
            })
        if command.startswith("p "):
            data_manager_parent_conn.send({
                "key": KEYS.PLAY_RECORDING,
                "name": command[2:]
            })

    data_manager_parent_conn.send({
        "key": KEYS.EXIT
    })

    print("Waiting to join processes")
    data_manager.join()
    print("Joined processes")



if __name__ == "__main__":
    main()


