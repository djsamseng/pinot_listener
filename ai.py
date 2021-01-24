'''
Each node has state

i1
i2
i3    * input weights (input conn stregth)
i4
i5

->  Node value = in * input weights - on * output weights
    Only outputs the value that is > 0.5 so stuff builds up then it starts
    outputting. The node can't accept anything if its maxed out, the higher
    the nodes value the less it can take in which forces the signal to go
    to other nodes


-> * output weights

->
o1
o2
o3
o4
o5
'''

import numpy as np

NUM_INPUTS = 2048
MAX_VAL = 255

NODES_PER_GROUP = 100
NODE_THRESHOLD = 0.5

def single_node_example():
    '''
    Need to add input resistance into the picture. The higher t
    '''
    this_nodes_orig_value = 0.3

    inputs_to_this_node = np.random.rand(1, NODES_PER_GROUP)
    input_weights_to_this_node = np.random.rand(NODES_PER_GROUP, 1) / NODES_PER_GROUP
    this_nodes_value_change = np.dot(inputs_to_this_node, input_weights_to_this_node)

    if this_nodes_orig_value + this_nodes_value_change > NODE_THRESHOLD:
        error_message = '''Super charged node
            current value:{0} value change: {1}'''.format(this_nodes_orig_value, this_nodes_value_change)
        # assert(False, error_message)
        print(error_message)
    this_nodes_mid_value = this_nodes_orig_value + this_nodes_value_change

    this_nodes_value_to_output = max(this_nodes_mid_value - NODE_THRESHOLD, 0)
    this_nodes_after_value = this_nodes_mid_value - this_nodes_value_to_output

    output_weights_from_this_node = np.random.rand(1, NODES_PER_GROUP) / NODES_PER_GROUP
    output_nodes_resistance = np.random.randn(1, NODES_PER_GROUP)
    output_nodes_resistance[output_nodes_resistance < 0] = output_nodes_resistance[output_nodes_resistance < 0] * -1

    resistence_multiplicative_effect = np.ones_like(output_nodes_resistance) - (output_nodes_resistance / np.max(output_nodes_resistance))

    output_weights_from_this_node_with_resistance = output_weights_from_this_node * resistence_multiplicative_effect

    outputs_from_this_node = np.dot(this_nodes_value_to_output, output_weights_from_this_node_with_resistance)
    outputs_from_this_node[outputs_from_this_node < 0] = 0

    LEARN_RATIO = 0.001
    LOG_LENGTH = 4
    input_weights_to_this_node_after_learning = input_weights_to_this_node + input_weights_to_this_node * np.rot90(inputs_to_this_node, k=-1) * LEARN_RATIO
    output_weights_from_this_node_after_learning = output_weights_from_this_node + output_weights_from_this_node * outputs_from_this_node * LEARN_RATIO

    print("Input values:", inputs_to_this_node[0][:LOG_LENGTH])
    print("Input weights before:", np.rot90(input_weights_to_this_node[:LOG_LENGTH]))
    print("Input weights learned:", np.rot90(input_weights_to_this_node_after_learning[:LOG_LENGTH]))
    print("Value state before:{0} increased:{1} outputted:{2} final:{3}".format(this_nodes_orig_value, this_nodes_value_change, this_nodes_value_to_output, this_nodes_after_value))
    print("Resistance:", output_nodes_resistance[0][:LOG_LENGTH])
    print("Output weights before:", output_weights_from_this_node[0][:LOG_LENGTH])
    print("Output weights considering resistance:", output_weights_from_this_node_with_resistance[0][0:LOG_LENGTH])
    print("Output weights learned:", output_weights_from_this_node_after_learning[0][:LOG_LENGTH])
    print("Output value:", outputs_from_this_node[0][:LOG_LENGTH])

if __name__ == "__main__":
    single_node_example()
