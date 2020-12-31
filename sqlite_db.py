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
        (id VARCHAR(50) PRIMARY KEY NOT NULL,
         inputting_node INT NOT NULL REFERENCES nodes(id),
         outputting_node INT NOT NULL REFERENCES nodes(id),
         weight INT NOT NULL);''')
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
            "input_connections": OrderedDict(),
        })
    conn.commit()
    existing_conn_ids = {}
    for i in range(num_nodes):
        for _ in range(num_connections_per_node):
            dest = random.randint(num_input_nodes, num_nodes - 1)
            random_weight = random.random() * 2 - 1
            conn_id = get_conn_id(dest, i)
            while dest is i or conn_id in existing_conn_ids:
                dest = random.randint(0, num_nodes - 1)
                conn_id = get_conn_id(dest, i)
            existing_conn_ids[conn_id] = True
            conn.execute('''INSERT INTO connections
                (id, inputting_node, outputting_node, weight)
                VALUES ("{0}", {1}, {2}, {3})'''.format(conn_id, dest, i, random_weight))
    conn.commit()

def get_conn_id(inputting_node_key, outputting_node_key):
    return "{0}->{1}".format(outputting_node_key, inputting_node_key)

def save_nodes(nodes):
    progress = 0
    num_nodes = len(nodes.keys())
    i = 0
    print("Saving...", flush=True)
    for inputting_node_key, node in nodes.items():
        j = 1
        if (inputting_node_key / num_nodes) >= progress:
            progress += 0.1
            print("Saving node: {0}/{1}".format(inputting_node_key, num_nodes), flush=True)
        conn.execute('''UPDATE nodes
            SET value={0}
            WHERE id={1}'''.format(node["value"], inputting_node_key))
        for outputting_node_key, connection in node["input_connections"].items():
            conn_id = get_conn_id(inputting_node_key, outputting_node_key)
            conn.execute('''UPDATE connections
                SET weight={0}
                WHERE id="{1}"
                '''.format(connection["weight"], conn_id))
            j += 1
        i += 1
    conn.commit()
    print("Saving finished", flush=True)

def init_worker_sqlite(worker_id, num_workers):
    num_nodes = conn.execute("SELECT COUNT(id) FROM nodes").fetchall()[0][0]
    owned_nodes = {}
    for i in range(num_nodes):
        if (i % num_workers) == worker_id:
            owned_nodes[i] = {
                "input_connections": {},
                "value": 0
            }
    sql_node_ids = "("
    sql_node_ids += ", ".join([str(a) for a in owned_nodes.keys()])
    sql_node_ids += ")"
    res = conn.execute('''SELECT
        nodes.id, nodes.value,
        connections.inputting_node, connections.outputting_node, connections.weight
        FROM nodes
        INNER JOIN connections
        ON nodes.id = connections.inputting_node
        WHERE nodes.id in {0}'''.format(sql_node_ids))
    for node in res.fetchall():
        node_id = node[0]
        node_value = node[1]
        inputting_node = node[2]
        outputting_node = node[3]
        connection_weight=  node[4]

        owned_nodes[node_id]["value"] = node_value
        owned_nodes[inputting_node]["input_connections"][outputting_node] = {
            "weight": connection_weight
        }
    for _, node in owned_nodes.items():
        node["input_connections"] = get_ordered_dict_by_key(node["input_connections"])
    return get_ordered_dict_by_key(owned_nodes)

def set_audio_input(node_id):
    conn = sqlite3.connect("pyaudio.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE nodes SET value = 1 WHERE nodes.id = {0}".format(node_id))
    cursor.execute('''UPDATE nodes
        SET value = 0
        WHERE 0 <= nodes.id AND nodes.id < 256 AND nodes.id IS NOT {0}'''.format(node_id))
    conn.commit()
