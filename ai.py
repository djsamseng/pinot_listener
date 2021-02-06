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
import unittest

from types import SimpleNamespace

NUM_INPUTS = 2048
MAX_VAL = 255

NODES_PER_GROUP = 200
NODE_CONNECTIVITY = 10
NODE_THRESHOLD = 0.5
RESISTANCE_TRESHOLD = 1
COPY_AND_ASSERT = True

LOG_LENGTH = 4
LOG_ACTIVATION = False
LEARN_RATIO = 0.001

def calc_single_node_incoming(inputs_to_this_node,
    input_weights_to_this_node,
    this_nodes_orig_value,
    node_threshold,
    resistance_threshold,
    learn_ratio):

    this_nodes_value_change = np.dot(inputs_to_this_node, input_weights_to_this_node)

    if this_nodes_orig_value + this_nodes_value_change > resistance_threshold:
        error_message = '''Node exceeded max threshold
            current value:{0} value change: {1}'''.format(this_nodes_orig_value, this_nodes_value_change)
        assert(False, error_message)
    if this_nodes_orig_value + this_nodes_value_change > NODE_THRESHOLD:
        if LOG_ACTIVATION:
            msg = "Node activated current value:{0} value change: {1}".format(this_nodes_orig_value, this_nodes_value_change)
            print(msg)

    this_nodes_mid_value = this_nodes_orig_value + this_nodes_value_change

    input_weights_to_this_node_after_learning = input_weights_to_this_node + input_weights_to_this_node * np.rot90(inputs_to_this_node, k=-1) * learn_ratio

    return (this_nodes_value_change, this_nodes_mid_value, input_weights_to_this_node_after_learning)

def calc_single_node_outgoing(this_nodes_value_to_output,
    this_nodes_after_value,
    output_weights_from_this_node,
    output_nodes_resistance,
    resistance_threshold,
    learn_ratio
    ):
    value_distributed = np.dot(this_nodes_value_to_output, output_weights_from_this_node) / np.sum(output_weights_from_this_node)
    assert(np.allclose(np.sum(value_distributed), this_nodes_value_to_output))
    assert(np.all(output_nodes_resistance >= 0))
    assert(np.all(output_nodes_resistance <= resistance_threshold))
    new_output_nodes_values = value_distributed + output_nodes_resistance
    output_values_over_threshold = new_output_nodes_values - resistance_threshold
    output_values_over_threshold[output_values_over_threshold < 0] = 0
    value_distributed = value_distributed - output_values_over_threshold
    this_nodes_after_value = this_nodes_after_value + np.sum(output_values_over_threshold)

    output_weights_from_this_node_after_learning = output_weights_from_this_node + output_weights_from_this_node * value_distributed * learn_ratio

    return (this_nodes_after_value, value_distributed, output_weights_from_this_node_after_learning)

def calc_single_node(inputs_to_this_node,
    input_weights_to_this_node,
    this_nodes_orig_value,
    output_weights_from_this_node,
    output_nodes_resistance,
    node_connectivity,
    node_threshold,
    resistance_threshold,
    learn_ratio):

    assert (inputs_to_this_node.shape == (1, node_connectivity)), "Bad shape:" + str(inputs_to_this_node.shape) + "!=" + str((1, node_connectivity))
    assert (input_weights_to_this_node.shape == (node_connectivity, 1)), "Bad shape:" + str(input_weights_to_this_node.shape) + "!=" + str((node_connectivity, 1))
    assert (this_nodes_orig_value.shape == (1,1)), "Bad shape:" + str(this_nodes_orig_value.shape) + "!=(1,1)"
    assert (type(this_nodes_orig_value[0,0])) == np.float64, "Incorrect node value type:" + str(type(this_nodes_orig_value[0,0]))
    assert (output_weights_from_this_node.shape == (1, node_connectivity)), "Incorrect output weights shape:" + str(output_weights_from_this_node.shape) + "!=" + str((1, node_connectivity))
    assert (output_nodes_resistance.shape == (1, node_connectivity)), "Incorrect output_nodes_resistance shape:" + str(output_nodes_resistance.shape) + "!=" + str((1, node_connectivity))

    (this_nodes_value_change, this_nodes_mid_value, input_weights_to_this_node_after_learning) = calc_single_node_incoming(inputs_to_this_node=inputs_to_this_node,
        input_weights_to_this_node=input_weights_to_this_node,
        this_nodes_orig_value=this_nodes_orig_value,
        node_threshold=node_threshold,
        resistance_threshold=resistance_threshold,
        learn_ratio=learn_ratio)

    # This value needs to be distributed, not multiplied by the output weights
    # Distribute to each connection such that we can't send more than RESISTANCE_THRESHOLD to each connection
    this_nodes_value_to_output = max(this_nodes_mid_value - node_threshold, 0)
    this_nodes_after_value = this_nodes_mid_value - this_nodes_value_to_output

    (this_nodes_after_value, value_distributed, output_weights_from_this_node_after_learning) = calc_single_node_outgoing(this_nodes_value_to_output=this_nodes_value_to_output,
        this_nodes_after_value=this_nodes_after_value,
        output_weights_from_this_node=output_weights_from_this_node,
        output_nodes_resistance=output_nodes_resistance,
        resistance_threshold=resistance_threshold,
        learn_ratio=learn_ratio)

    assert(np.allclose(this_nodes_orig_value + this_nodes_value_change, this_nodes_after_value + np.sum(value_distributed)))

    if this_nodes_after_value[0,0] > resistance_threshold:
        # This may be because resistance is high and input signal is high enough there's nowhere to go
        this_nodes_after_value[0,0] = resistance_threshold

    assert (value_distributed.shape == (1, node_connectivity))

    return {
        "input_weights_to_this_node_after_learning": input_weights_to_this_node_after_learning,
        "this_nodes_value_change": this_nodes_value_change,
        "this_nodes_value_to_output": this_nodes_value_to_output,
        "this_nodes_after_value": this_nodes_after_value,
        "output_weights_from_this_node_after_learning": output_weights_from_this_node_after_learning,
        "outputs_from_this_node": value_distributed
    }

def test_calc_single_node_1():
    this_nodes_orig_value = np.array([[0.3]])
    input_weights_to_this_node = np.array([
        [ 0.5 ],
        [ 0.1 ]
    ])
    output_weights_from_this_node = np.array([
        [ 0.2, 0.1 ]
    ])
    inputs_to_this_node = np.array([
        [ 0.2, 0.3 ],
    ])
    output_nodes_resistance = np.array([
        [ 0.0, 0.0 ],
    ])
    node_threshold = 0.5
    after = calc_single_node(inputs_to_this_node=inputs_to_this_node,
        input_weights_to_this_node=input_weights_to_this_node,
        this_nodes_orig_value=this_nodes_orig_value,
        output_weights_from_this_node=output_weights_from_this_node,
        output_nodes_resistance=output_nodes_resistance,
        node_connectivity=2,
        node_threshold=node_threshold,
        resistance_threshold=1,
        learn_ratio=0.1
    )
    after = SimpleNamespace(**after)
    expected_value_after = this_nodes_orig_value
    expected_value_after += input_weights_to_this_node[0,0] * inputs_to_this_node[0,0]
    expected_value_after += input_weights_to_this_node[1,0] * inputs_to_this_node[0,1]
    np.testing.assert_allclose(after.this_nodes_after_value, expected_value_after)
    np.testing.assert_allclose(after.this_nodes_after_value, np.array([[0.43]]))
    output_value = max(expected_value_after[0, 0] - node_threshold, 0)
    expected_outputs = np.array([[
        output_value * output_weights_from_this_node[0, 0] / np.sum(output_weights_from_this_node),
        output_value * output_weights_from_this_node[0, 1] / np.sum(output_weights_from_this_node)
    ]])
    np.testing.assert_allclose(after.outputs_from_this_node, expected_outputs)
    np.testing.assert_allclose(after.outputs_from_this_node, np.array([[0, 0]]))
    print("Unit tests 3 success")

def test_calc_single_node_2():
    this_nodes_orig_value = np.array([[0.3]])
    input_weights_to_this_node = np.array([
        [ 0.5 ],
        [ 0.1 ]
    ])
    output_weights_from_this_node = np.array([
        [ 0.2, 0.1 ]
    ])
    inputs_to_this_node = np.array([
        [ 0.2, 0.3 ],
    ])
    output_nodes_resistance = np.array([
        [ 0.0, 0.0 ],
    ])
    node_threshold = 0.2
    after = calc_single_node(inputs_to_this_node=inputs_to_this_node,
        input_weights_to_this_node=input_weights_to_this_node,
        this_nodes_orig_value=this_nodes_orig_value,
        output_weights_from_this_node=output_weights_from_this_node,
        output_nodes_resistance=output_nodes_resistance,
        node_connectivity=2,
        node_threshold=node_threshold,
        resistance_threshold=1,
        learn_ratio=0.1
    )
    after = SimpleNamespace(**after)
    expected_value_mid = this_nodes_orig_value
    expected_value_mid += input_weights_to_this_node[0,0] * inputs_to_this_node[0,0]
    expected_value_mid += input_weights_to_this_node[1,0] * inputs_to_this_node[0,1]
    # Over threshold
    expected_value_after = np.array([[node_threshold]])
    output_value = expected_value_mid[0, 0] - node_threshold
    np.testing.assert_allclose(after.this_nodes_after_value, expected_value_after)
    np.testing.assert_allclose(after.this_nodes_after_value, np.array([[0.2]]))

    expected_outputs = np.array([[
        output_value * output_weights_from_this_node[0, 0] / np.sum(output_weights_from_this_node),
        output_value * output_weights_from_this_node[0, 1] / np.sum(output_weights_from_this_node)
    ]])
    np.testing.assert_allclose(after.outputs_from_this_node, expected_outputs)
    np.testing.assert_allclose(after.outputs_from_this_node, np.array([[0.153333, 0.076667]]), 0.00001)
    print("Unit tests 2 success")

def single_node_example():
    '''
    Need to add input resistance into the picture. The higher t
    '''
    this_nodes_orig_value = np.array([[0.3]])

    input_weights_to_this_node = np.random.rand(NODE_CONNECTIVITY, 1) / NODE_CONNECTIVITY
    output_weights_from_this_node = np.random.rand(1, NODE_CONNECTIVITY) / NODE_CONNECTIVITY

    itr_input_weights_to_this_node = input_weights_to_this_node
    itr_output_weights_from_this_node = output_weights_from_this_node
    itr_this_nodes_value = this_nodes_orig_value

    for i in range(10):
        inputs_to_this_node = np.random.rand(1, NODE_CONNECTIVITY)
        output_nodes_resistance = np.random.randn(1, NODE_CONNECTIVITY)
        output_nodes_resistance[output_nodes_resistance < 0] = output_nodes_resistance[output_nodes_resistance < 0] * -1
        output_nodes_resistance[output_nodes_resistance > RESISTANCE_TRESHOLD] = RESISTANCE_TRESHOLD
        after = calc_single_node(inputs_to_this_node=inputs_to_this_node,
            input_weights_to_this_node=itr_input_weights_to_this_node,
            this_nodes_orig_value=itr_this_nodes_value,
            output_weights_from_this_node=itr_output_weights_from_this_node,
            output_nodes_resistance=output_nodes_resistance,
            node_connectivity=NODE_CONNECTIVITY,
            node_threshold=NODE_THRESHOLD,
            resistance_threshold=RESISTANCE_TRESHOLD,
            learn_ratio=LEARN_RATIO)
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

def multi_node_example():
    orig_values = np.zeros((1, NODES_PER_GROUP))
    orig_values[:,:] = 0.3
    input_weights = np.random.rand(NODE_CONNECTIVITY, NODES_PER_GROUP) / NODE_CONNECTIVITY
    output_weights = np.random.rand(NODES_PER_GROUP, NODE_CONNECTIVITY) / NODE_CONNECTIVITY

    itr_values = orig_values
    itr_input_weights = input_weights
    itr_output_weights = output_weights

    for itr in range(10):
        itr_orig_values = np.copy(itr_values)
        itr_orig_input_weights = np.copy(itr_input_weights)
        itr_orig_output_weights = np.copy(itr_output_weights)
        for node_id in range(NODE_CONNECTIVITY + 1, NODES_PER_GROUP - NODE_CONNECTIVITY):
            input_start_idx = node_id - NODE_CONNECTIVITY - 1
            input_end_idx = node_id -1
            inputs_to_this_node = itr_orig_values[0:1, input_start_idx:input_end_idx]
            assert(inputs_to_this_node.shape == (1, NODE_CONNECTIVITY))
            input_weights_to_this_node = itr_orig_input_weights[:, node_id:node_id+1]
            output_weights_from_this_node = itr_orig_output_weights[node_id:node_id+1, :]
            output_start_idx = node_id + 1
            output_end_idx = node_id + NODE_CONNECTIVITY + 1
            output_nodes_resistance = itr_orig_values[0:1, output_start_idx:output_end_idx]
            assert(output_nodes_resistance.shape == (1, NODE_CONNECTIVITY))

            after = calc_single_node(inputs_to_this_node=inputs_to_this_node,
                input_weights_to_this_node=input_weights_to_this_node,
                this_nodes_orig_value=itr_orig_values[0:1, node_id:node_id+1],
                output_weights_from_this_node=output_weights_from_this_node,
                output_nodes_resistance=output_nodes_resistance,
                node_connectivity=NODE_CONNECTIVITY,
                node_threshold=NODE_THRESHOLD,
                resistance_threshold=RESISTANCE_TRESHOLD,
                learn_ratio=LEARN_RATIO)
            after = SimpleNamespace(**after)

            itr_input_weights[:, node_id:node_id+1] = after.input_weights_to_this_node_after_learning
            itr_values[0,node_id] = after.this_nodes_after_value[0,0]
            itr_output_weights[node_id+1, :] = after.output_weights_from_this_node_after_learning


        print("MultiNode itr:{0}".format(itr))

        assert(not np.allclose(itr_orig_input_weights, itr_input_weights))
        assert(not np.allclose(itr_orig_output_weights, itr_output_weights))




if __name__ == "__main__":
    single_node_example()
    multi_node_example()
    test_calc_single_node_1()
    test_calc_single_node_2()
