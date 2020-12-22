import numpy as np
import audio

# Get data from db
# Master transfers
# Each GPU holds state


def get_a_from_mem(mem):
    a = mem[0]
    # mem[0][4] is now a ref to mem[1][2]
    # writing mem[0][4] sets it to a new value thus does not set mem[1][2]
    # changing mem[1][2] changes mem[0][4]
    a[4] = mem[1][2]
    return a

def get_b_from_mem(mem):
    b = mem[1]
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
    b[2] = 3
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

def main():
    demo_slice_ref()
    audio.record(audio_callback)


if __name__ == "__main__":
    main()

