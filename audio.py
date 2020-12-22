import pyaudio
import time

from functools import partial

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 0.1

def callback(in_data, frame_count, time_info, flag, audio_callback):
    # len 2048 each element is an int from [0, 255]
    audio_callback(in_data)
    return in_data, pyaudio.paComplete

def record(audio_callback):
    p = pyaudio.PyAudio()

    bound_callback = partial(callback, audio_callback=audio_callback)
    stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=bound_callback)
    print("Start recording", flush=True)
    stream.start_stream()

    while stream.is_active():
        # stream.stop_stream()
        pass
    print("Done recording", flush=True)

    stream.close()
    p.terminate()

