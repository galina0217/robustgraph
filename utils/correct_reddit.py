import json


if __name__ == '__main__':
    prefix = '../data/reddit/reddit'
    G_data = json.load(open(prefix + "-G_ori.json"))
    id_map = json.load(open(prefix + "-id_map_ori.json"))
    class_map = json.load(open(prefix + "-class_map_ori.json"))

    for id_key in list(class_map.keys()):
        replace_value = class_map.pop(id_key)
        class_map[str(id_map[id_key])] = replace_value

    with open(prefix + "-class_map.json", 'w') as f:
        f.write(json.dumps(class_map))
        f.close()

    check_nodes = [False for _ in range(len(id_map))]
    nodes = G_data['nodes']
    for node in nodes:
        node['id'] = int(id_map[node['id']])
        check_nodes[node['id']] = True

    for id_key in list(id_map.keys()):
        replace_value = int(id_map.pop(id_key))
        id_map[str(replace_value)] = replace_value
        if not check_nodes[replace_value]:
            nodes.append({'id': replace_value})
    G_data['nodes'] = nodes

    with open(prefix + "-G.json", 'w') as f:
        f.write(json.dumps(G_data))
        f.close()

    with open(prefix + "-id_map.json", 'w') as f:
        f.write(json.dumps(id_map))
        f.close()
