from autoqchem_local.api.api import AutoChem

ctrl = AutoChem()
ctrl.generate_gjf_files(smiles_file='./smiles.smi')