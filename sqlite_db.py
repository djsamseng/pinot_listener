import random
import sqlite3

conn = sqlite3.connect("pyaudio.db")

def create_new_node_network_sqlite():
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
            "_id": str(i),
            "output_connections": [],
            "input_connections": [],
        })
    conn.commit()
    conn_id = 0
    for i in range(num_nodes):
        for j in range(num_connections_per_node):
            dest = random.randint(0, num_nodes - 1)
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
            owned_nodes[i] = { "output_connections": {} }
    sql_node_ids = "("
    sql_node_ids += ", ".join([str(a) for a in owned_nodes.keys()])
    sql_node_ids += ")"
    res = conn.execute('''SELECT
        nodes.id, connections.input_node, connections.output_node
        FROM nodes
        INNER JOIN connections
        ON nodes.id = connections.input_node
        WHERE nodes.id in {0}'''.format(sql_node_ids))
    for node in res.fetchall():
        node_id = node[0]
        input_node = node[1]
        output_node = node[2]
        owned_nodes[node_id]["output_connections"][output_node] = {}
    return owned_nodes

def set_audio_input(node_id):
    conn = sqlite3.connect("pyaudio.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE nodes SET value = 1 WHERE nodes.id = {0}".format(node_id))
    cursor.execute('''UPDATE nodes
        SET value = 0
        WHERE 0 <= nodes.id AND nodes.id < 256 AND nodes.id IS NOT {0}'''.format(node_id))
    conn.commit()
