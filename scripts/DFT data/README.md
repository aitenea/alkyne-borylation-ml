# DFT calculation
The provided [dataset](https://github.com/aitenea/alkyne-borylation-ml/blob/main/data/df_bor_full.csv) that we used to train and validate our machine learning models was generated with DFT calculations using both Gaussian and Morfeus to obtain the steric and electrical descriptors of both ligands and reactives. 

We generated the _.gjf_ files that Gaussian requires, extracted the _.log_ results, performed the Morfeus calculations and generated the final _.csv_ dataset with an adapted version of [Auto-QChem](https://github.com/dkesada/auto-qchem_exp).