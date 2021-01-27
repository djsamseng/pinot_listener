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

from types import SimpleNamespace

NUM_INPUTS = 2048
MAX_VAL = 255

NODES_PER_GROUP = 100
NODE_THRESHOLD = 0.5
RESISTANCE_TRESHOLD = 1

LOG_LENGTH = 4
LEARN_RATIO = 0.001

def calc_single_node(inputs_to_this_node,
    input_weights_to_this_node,
    this_nodes_orig_value,
    output_weights_from_this_node,
    output_nodes_resistance):



    this_nodes_value_change = np.dot(inputs_to_this_node, input_weights_to_this_node)

    if this_nodes_orig_value + this_nodes_value_change > RESISTANCE_TRESHOLD:
        error_message = '''Node exceeded max threshold
            current value:{0} value change: {1}'''.format(this_nodes_orig_value, this_nodes_value_change)
        assert(False, error_message)
    if this_nodes_orig_value + this_nodes_value_change > NODE_THRESHOLD:
        msg = "Node activated current value:{0} value change: {1}".format(this_nodes_orig_value, this_nodes_value_change)
        print(msg)

    this_nodes_mid_value = this_nodes_orig_value + this_nodes_value_change

    # This value needs to be distributed, not multiplied by the output weights
    # Distribute to each connection such that we can't send more than RESISTANCE_THRESHOLD to each connection
    this_nodes_value_to_output = max(this_nodes_mid_value - NODE_THRESHOLD, 0)
    this_nodes_after_value = this_nodes_mid_value - this_nodes_value_to_output

    value_distributed = np.dot(this_nodes_value_to_output, output_weights_from_this_node) / np.sum(output_weights_from_this_node)
    assert(np.allclose(np.sum(value_distributed), this_nodes_value_to_output))
    assert(np.all(output_nodes_resistance >= 0))
    assert(np.all(output_nodes_resistance <= RESISTANCE_TRESHOLD))
    new_output_nodes_values = value_distributed + output_nodes_resistance
    output_values_over_threshold = new_output_nodes_values - RESISTANCE_TRESHOLD
    output_values_over_threshold[output_values_over_threshold < 0] = 0
    value_distributed = value_distributed - output_values_over_threshold
    this_nodes_after_value = this_nodes_after_value + np.sum(output_values_over_threshold)

    input_weights_to_this_node_after_learning = input_weights_to_this_node + input_weights_to_this_node * np.rot90(inputs_to_this_node, k=-1) * LEARN_RATIO
    output_weights_from_this_node_after_learning = output_weights_from_this_node + output_weights_from_this_node * value_distributed * LEARN_RATIO

    assert(np.allclose(this_nodes_orig_value + this_nodes_value_change, this_nodes_after_value + np.sum(value_distributed)))

    return {
        "input_weights_to_this_node_after_learning": input_weights_to_this_node_after_learning,
        "this_nodes_value_change": this_nodes_value_change,
        "this_nodes_value_to_output": this_nodes_value_to_output,
        "this_nodes_after_value": this_nodes_after_value,
        "output_weights_from_this_node_after_learning": output_weights_from_this_node_after_learning,
        "outputs_from_this_node": value_distributed
    }

def single_node_example():
    '''
    Need to add input resistance into the picture. The higher t
    '''
    this_nodes_orig_value = 0.3

    input_weights_to_this_node = np.random.rand(NODES_PER_GROUP, 1) / NODES_PER_GROUP
    output_weights_from_this_node = np.random.rand(1, NODES_PER_GROUP) / NODES_PER_GROUP

    itr_input_weights_to_this_node = input_weights_to_this_node
    itr_output_weights_from_this_node = output_weights_from_this_node
    itr_this_nodes_value = this_nodes_orig_value

    for i in range(10):
        inputs_to_this_node = np.random.rand(1, NODES_PER_GROUP)
        output_nodes_resistance = np.random.randn(1, NODES_PER_GROUP)
        output_nodes_resistance[output_nodes_resistance < 0] = output_nodes_resistance[output_nodes_resistance < 0] * -1
        output_nodes_resistance[output_nodes_resistance > RESISTANCE_TRESHOLD] = RESISTANCE_TRESHOLD
        after = calc_single_node(inputs_to_this_node=inputs_to_this_node,
            input_weights_to_this_node=itr_input_weights_to_this_node,
            this_nodes_orig_value=itr_this_nodes_value,
            output_weights_from_this_node=itr_output_weights_from_this_node,
            output_nodes_resistance=output_nodes_resistance)
        after = SimpleNamespace(**after)

        itr_input_weights_to_this_node = after.input_weights_to_this_node_after_learning
        itr_this_nodes_value = after.this_nodes_after_value
        itr_output_weights_from_this_node = after.output_weights_from_this_node_after_learning

        print("Itr:{0} value:{1} sent:{2}".format(i, after.this_nodes_after_value, np.sum(after.outputs_from_this_node)))

    print("\n")
    print("Input values:", inputs_to_this_node[0][:LOG_LENGTH])
    print("Input weights before:", np.rot90(input_weights_to_this_node[:LOG_LENGTH]))
    print("Input weights learned:", np.rot90(after.input_weights_to_this_node_after_learning[:LOG_LENGTH]))
    print("Value state before:{0} increased:{1} outputted:{2} final:{3}".format(this_nodes_orig_value,
        after.this_nodes_value_change,
        after.this_nodes_value_to_output,
        after.this_nodes_after_value))
    print("Resistance:", output_nodes_resistance[0][:LOG_LENGTH])
    print("Output weights before:", output_weights_from_this_node[0][:LOG_LENGTH])
    print("Output weights learned:", after.output_weights_from_this_node_after_learning[0][:LOG_LENGTH])
    print("Output value:", after.outputs_from_this_node[0][:LOG_LENGTH])


if __name__ == "__main__":
    single_node_example()
