import json


def read_json(p):
    with open(p, 'r') as fp:
        json_contents = json.load(fp)
    return json_contents


def write_json(data, p):
    with open(p, 'w') as fp:
        json.dump(data, fp, indent=2)