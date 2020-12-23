import random
import sqlite3

from collections import OrderedDict

conn = sqlite3.connect("pyaudio.db")

def get_ordered_dict_by_key(obj):
    ordered_keys = list(obj.keys())
    ordered_keys.sort()
    ordered_dict = OrderedDict()
    for key in ordered_keys:
        ordered_dict[key] = obj[key]
    return ordered_dict

def create_new_node_network_sqlite(num_input_nodes=2048):
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.execute("DROP TABLE IF EXISTS connections")
    conn.execute('''CREATE TABLE nodes
        (id INT PRIMARY KEY NOT NULL,
         value INT NOT NULL);''')
    conn.execute('''CREATE TABLE connections
        (id INT PRIMARY KEY NOT NULL,
         input_node INT NOT NULL REFERENCES nodes(id),
         output_node INT NOT NULL REFERENCES nodes(id));''')
    conn.commit()
    print("Created sqlite tables", flush=True)
    num_connections_per_node = 10
    num_nodes = 10 * 1000
    new_nodes = []
    for i in range(num_nodes):
        conn.execute('''INSERT INTO nodes
            (id, value)
            VALUES({0}, 0)'''.format(i))
        new_nodes.append({
            "id": i,
            "output_connections": OrderedDict(),
            "input_connections": OrderedDict(),
        })
    conn.commit()
    conn_id = 0
    for i in range(num_nodes):
        for _ in range(num_connections_per_node):
            dest = random.randint(num_input_nodes, num_nodes - 1)
            while dest is i:
                dest = random.randint(0, num_nodes - 1)
            conn.execute('''INSERT INTO connections
                (id, input_node, output_node)
                VALUES ({0}, {1}, {2})'''.format(conn_id, i, dest))
            conn_id += 1
    conn.commit()

def init_worker_sqlite(worker_id, num_workers):
    num_nodes = conn.execute("SELECT COUNT(id) FROM nodes").fetchall()[0][0]
    owned_nodes = {}
    for i in range(num_nodes):
        if (i % num_workers) == worker_id:
            owned_nodes[i] = {
                "input_connections": {},
                "output_connections": {}
            }
    sql_node_ids = "("
    sql_node_ids += ", ".join([str(a) for a in owned_nodes.keys()])
    sql_node_ids += ")"
    res = conn.execute('''SELECT
        nodes.id, nodes.value, connections.input_node, connections.output_node
        FROM nodes
        INNER JOIN connections
        ON nodes.id = connections.input_node
        WHERE nodes.id in {0}'''.format(sql_node_ids))
    for node in res.fetchall():
        node_id = node[0]
        node_value = node[1]
        input_node = node[2]
        output_node = node[3]

        owned_nodes[node_id]["value"] = node_value
        owned_nodes[output_node]["input_connections"][input_node] = {}
        owned_nodes[node_id]["output_connections"][output_node] = {}
    for _, node in owned_nodes.items():
        node["input_connections"] = get_ordered_dict_by_key(node["input_connections"])
        node["output_connections"] = get_ordered_dict_by_key(node["output_connections"])
    return get_ordered_dict_by_key(owned_nodes)

def set_audio_input(node_id):
    conn = sqlite3.connect("pyaudio.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE nodes SET value = 1 WHERE nodes.id = {0}".format(node_id))
    cursor.execute('''UPDATE nodes
        SET value = 0
        WHERE 0 <= nodes.id AND nodes.id < 256 AND nodes.id IS NOT {0}'''.format(node_id))
    conn.commit()
