import pyaudio
import time
import wave

from functools import partial

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 0.1

def callback(in_data, frame_count, time_info, flag, audio_record_queue):
    # len 2048 each element is an int from [0, 255]
    audio_record_queue.put(in_data)
    return in_data, pyaudio.paContinue

def save_recording(frames, filename):
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print(len(frames))
    for frame in frames:
        wf.writeframes(frame)
    wf.close()

def record(audio_child_conn, audio_record_queue):
    p = pyaudio.PyAudio()

    bound_callback = partial(callback, audio_record_queue=audio_record_queue)
    stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=bound_callback)
    print("Start recording", flush=True)
    stream.start_stream()

    play_stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK)

    while stream.is_active():
        has_message = audio_child_conn.poll(0.01)
        if has_message:
            msg = audio_child_conn.recv()
            # print("Received! {0}".format(msg))
            if msg and msg["key"] == "exit":
                # If we don't close this before stopping the stream the process will hang
                print("Closing audio queue and stream", flush=True)
                stream.stop_stream()
                audio_record_queue.close()
                audio_child_conn.close()
                break
            if msg and msg["key"] == "play_audio":
                play_stream.write(msg["data"])

    print("Done recording", flush=True)

    stream.close()
    play_stream.close()
    p.terminate()

    audio_child_conn.close()

    print("Closed audio process", flush=True)

