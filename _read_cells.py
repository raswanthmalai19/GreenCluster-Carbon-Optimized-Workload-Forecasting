import json, sys

nb_file = sys.argv[1]
targets = sys.argv[2:]

with open(nb_file) as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    cid = cell.get('id', '')
    for t in targets:
        if t in cid:
            print(f'=== Cell {cid} (idx {i}) ===')
            print(''.join(cell.get('source', [])))
            print()
