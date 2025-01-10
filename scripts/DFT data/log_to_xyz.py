from openbabel import pybel
from pathlib import Path

output = './xyz_coords/'
Path(output).mkdir(parents=True, exist_ok=True)

p = Path('./').glob('**/*')
files = [x for x in p if x.is_file()]

for file in files:
    molecule = next(pybel.readfile('log', file.__str__()))
    file_name = file.__str__()[:-8]
    smiles = molecule.write('smiles').split('\t')[0]
    data = molecule.write('xyz').split('\n')
    data[1] = smiles
    data = '\n'.join(data)
    with open(output + file_name + '.xyz', "w") as f:
        f.write(data)
