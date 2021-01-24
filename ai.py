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
    this_nodes_current_value = 0.3
    this_nodes_resistance_before = this_nodes_current_value

    inputs_to_this_node = np.random.rand(1, NODES_PER_GROUP)
    input_weights_to_this_node = np.random.rand(NODES_PER_GROUP, 1) / NODES_PER_GROUP
    this_nodes_value_change = np.dot(inputs_to_this_node, input_weights_to_this_node)

    if this_nodes_current_value + this_nodes_value_change > NODE_THRESHOLD:
        error_message = '''Super charged node
            current value:{0} value change: {1}'''.format(this_nodes_current_value, this_nodes_value_change)
        # assert(False, error_message)
        print(error_message)
    this_nodes_current_value += this_nodes_value_change

    this_nodes_value_to_output = max(this_nodes_current_value - NODE_THRESHOLD, 0)
    this_nodes_current_value -= this_nodes_value_to_output
    print("Final node value:{0} after outputting:{1}".format(this_nodes_current_value, this_nodes_value_to_output))

    output_weights_from_this_node = np.random.rand(1, NODES_PER_GROUP) / NODES_PER_GROUP
    output_nodes_resistance = np.random.randn(1, NODES_PER_GROUP)
    output_nodes_resistance[output_nodes_resistance < 0] = output_nodes_resistance[output_nodes_resistance < 0] * -1
    print("Resistance:", output_nodes_resistance[0][:10])
    resistence_multiplicative_effect = np.ones_like(output_nodes_resistance) - (output_nodes_resistance / np.max(output_nodes_resistance))
    print("Weights before:", output_weights_from_this_node[0][:10])
    output_weights_from_this_node *= resistence_multiplicative_effect
    print("Weights after:", output_weights_from_this_node[0][0:10])

    # output weights need to be changed to consider input resistance
    outputs_from_this_node = np.dot(this_nodes_value_to_output, output_weights_from_this_node)
    outputs_from_this_node[outputs_from_this_node < 0] = 0
    print("Final outputs:", outputs_from_this_node[0][:10])
    this_nodes_resistance_after = this_nodes_current_value

if __name__ == "__main__":
    single_node_example()
