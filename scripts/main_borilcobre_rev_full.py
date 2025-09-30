# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import logging
from datetime import datetime
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, OrdinalEncoder
from sklearn.decomposition import PCA
from mlworkflow.models.regression_models import LinRegModel, RidgeModel, RFModel, SVRModel, ElasticModel, LassoModel, GaussianProcessModel
from mlworkflow.models.classif_models import LogRegModel, RFCModel, SVCModel
from mlworkflow.models.greedy_var_sel import GreedyVarSel, BackwardsGreedyVarSel
from mlworkflow.models.genetic_var_sel import GeneticVarSel
from mlworkflow.models.metrics import Metrics
from mlworkflow.models.diff_ev_optim import DiffEvOptim
from mlworkflow.utils import find_k_tani
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import r2_score
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def error_plot(orig, preds, label, title="Leave-one-out crossvalidation results"):
    _, ax = plt.subplots(figsize=(5, 5))
    display = PredictionErrorDisplay.from_predictions(
        preds, orig, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.6}
    )
    ax.set_title(title, size=12)
    ax.set_xlabel(f'{label} predicted values', size=12)
    ax.set_ylabel(f'Real values', size=12)
    # for name, score in scores.items():
    #    ax.plot([], [], " ", label=f"{name}: {score}")
    # ax.get_legend().remove()
    # ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    # plt.savefig("yield_pred.pdf")


def eval_loo_model_param(df, obj_var, feature_var, subset, solvent):
    df = df[list(map(lambda x: x in subset, df['solubility']))]
    df = df[df['solvent'] == solvent]

    m = SVRModel(components=[MinMaxScaler()], restrict=True)

    metrics = Metrics(plot_res=False)
    compounds = [''] * len(df)
    res, preds = m.eval_loo(df, obj_var, feature_var, metrics, compounds, print_res=False, plot_res=False)

    return df.index, preds


def remove_str_cols(df):
    """ Remove the columns that have strings in their first row in the dataframe """
    idx = [not isinstance(df[x][0], str) for x in df.columns]

    return df.loc[:, idx]


def train_and_save(df_path):
    df = pd.read_csv(df_path)
    # df.dropna(axis='rows', inplace=True)
    df.loc[df['ratio'] < -2.2, 'ratio'] = df.loc[df['ratio'] < -2.2, 'ratio'] + np.random.normal(0, 0.1, len(df.loc[df['ratio'] < -2.2, 'ratio']))
    # df = df.select_dtypes(exclude=['object'])  # Remove columns with strings

    obj_var = ["ratio"]

    feature_var = ['Solid_angle_G_Emin_ligand', 'nucleofug_Boltz_ligand', 'IP_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'nucleo_Emin_react', 'Dispersion_area_Boltz_ligand', 'Pyr_P_Boltz_react', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'VBur_Emin_ligand', 'Sterimol_bur_B_5_Emin_react', 'Dispersion_area_Emin_ligand', 'Solid_angle_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_Emin_ligand', 'Pyr_alpha_Emin_react', 'Solid_angle_G_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'HOMO_Boltz_react']

    logging.info('Training and saving model with all data')
    m = SVRModel(components=[MinMaxScaler()])
    m.assign_params([3.0, 0.61906065, 0.0, 1.0, 0.1])

    # m.assign_params(params)
    m.fit(df, obj_var, feature_var, split=False, print_res=False)
    m.save_model()
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))


def eval_model(df_path):
    df = pd.read_csv(df_path)
    df.dropna(axis='rows', inplace=True)
    # df = df.sample(frac=1, random_state=42)  # Shuffle the dataset
    # Add white noise to the 98:2 and 99:1 extremes
    df.loc[df['ratio'] < -2.2, 'ratio'] = df.loc[df['ratio'] < -2.2, 'ratio'] + np.random.normal(0, 0.1, len(
        df.loc[df['ratio'] < -2.2, 'ratio']))

    df_val = df.sample(frac=0.1, random_state=43)
    df = df[[x not in df_val.index for x in df.index]]

    obj_var = ["ratio"]

    plot_res = False

    # SVR
    feature_var = ['Solid_angle_G_Emin_ligand', 'nucleofug_Boltz_ligand', 'IP_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'nucleo_Emin_react', 'Dispersion_area_Boltz_ligand', 'Pyr_P_Boltz_react', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'VBur_Emin_ligand', 'Sterimol_bur_B_5_Emin_react', 'Dispersion_area_Emin_ligand', 'Solid_angle_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_Emin_ligand', 'Pyr_alpha_Emin_react', 'Solid_angle_G_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'HOMO_Boltz_react']
    # GP
    feature_var = ['Solid_angle_G_Emin_ligand', 'VBur_Emin_react', 'electrofug_Emin_ligand', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'Dispersion_vol_Emin_react', 'VBur_Emin_ligand', 'Pyr_alpha_Emin_react', 'Sterimol_bur_B_1_Emin_ligand', 'Dispersion_vol_Boltz_react', 'Sterimol_bur_L_Boltz_ligand', 'Dispersion_area_Boltz_react', 'Sterimol_bur_L_Emin_ligand']
    # RF
    feature_var = ['Sterimol_B_1_Boltz_ligand', 'Sterimol_bur_L_Emin_react', 'Sterimol_bur_L_Boltz_react', 'Sterimol_bur_B_1_Boltz_ligand', 'Sterimol_L_Emin_ligand', 'nucleofug_Emin_react', 'nucleo_Boltz_ligand', 'Sasa_mol_VBur_Boltz_react', 'EA_Boltz_react', 'Solid_angle_G_Boltz_ligand', 'electro_Emin_react', 'Dispersion_vol_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'Sasa_mol_VBur_Emin_ligand', 'SASA_area_Emin_react', 'Sterimol_B_1_Emin_ligand', 'Sterimol_bur_B_5_Emin_ligand', 'dipole_Boltz_ligand', 'Free_VBur_Boltz_react', 'nucleo_Emin_ligand', 'SASA_area_Boltz_react', 'Sasa_distal_VBur_Emin_react']
    # Ridge
    feature_var = ['Free_VBur_Emin_ligand', 'LUMO_Emin_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'Sasa_mol_VBur_Emin_react', 'Sterimol_L_Emin_ligand', 'Free_VBur_Boltz_react', 'electro_Boltz_ligand', 'VBur_Emin_ligand', 'Pyr_P_angle_Emin_react', 'Sterimol_B_5_Boltz_ligand', 'EA_Boltz_ligand', 'Sterimol_L_Boltz_ligand', 'SASA_area_Boltz_ligand', 'Sterimol_B_5_Boltz_react', 'Sterimol_B_1_Emin_react', 'electro_Emin_ligand', 'nucleofug_Boltz_react', 'VBur_Boltz_react', 'Solid_angle_G_Emin_ligand', 'SASA_area_Emin_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'Dispersion_P_int_Boltz_ligand']

    m = RidgeModel(components=[MinMaxScaler()])
    metrics = Metrics(default_cat='f1', plot_res=plot_res)
    #m.assign_params([2.26195623, 0.67318304, 0.0, 0.99999949, 0.09999995])
    logging.info('Running model evaluation given a feature set')
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    res = m.eval_cv(df, obj_var, feature_var, metrics, shuffle=True, seed=42, folds=5, print_res=True)

    logging.info(f'Average error obtained: {res}')


def eval_loo_model(df_path):
    df = pd.read_csv(df_path)
    # df = df.sample(frac=1, random_state=42)  # Shuffle the dataset
    # Add white noise to the 98:2 and 99:1 extremes
    df.loc[df['ratio'] < -2.2, 'ratio'] = df.loc[df['ratio'] < -2.2, 'ratio'] + np.random.normal(0, 0.1, len(df.loc[df['ratio'] < -2.2, 'ratio']))
    # df.reset_index(inplace=True, drop=True)
    #df = df[df['ratio'] <= 2.0]
    # rdkit_lig = extract_rdkit_dataset(df['smiles_ligand'].values)
    # rdkit_lig.columns = [x + '_ligand' for x in rdkit_lig.columns]
    # df = pd.concat([df, rdkit_lig], axis=1)
    #df.dropna(axis='rows', inplace=True)
    #df = df.select_dtypes(exclude=['object'])  # Remove columns with strings
    idx = [131,  60,  95, 121,   2,  61,  36,  85, 101, 127,  45,  57,  19,
       120,   8,  89, 104, 124,  39,  26,  10,  87,  83,  49, 130, 116,
        20, 138,  69,  93, 103, 119, 102,  99, 137, 122,  38,  98,  43,
        41,  52,   6, 128,  97,  67,  80,   0,  86,  50,  73,  96,  17,
       113,  76,  29, 136,  22, 135,  15, 139,  21, 123,  72,  13,  37,
       134,  47,   5,  54,  28,  64, 125,  63,  65,  82,  14,   4, 117,
        35,  44, 133, 107, 118,  42, 109, 126,  16,  81,  56,  30,  23,
        25, 110,  24, 129,  55,  91,  84, 115,   1,  88,  53,  27, 100,
        92,  90,  68, 114,  70,  33,  18,  77,   7,  59,  32,  66,  51,
         9,  40,  48, 108,  78,  74,  58,  31,  75, 112,  46,   3,  79,
        12,  62, 132,  71, 106,  34,  11,  94, 105, 111]

    obj_var = ["ratio"]

    plot_res = True

    #feature_var = list(df.drop(obj_var, axis=1).columns)
    # feature_var = ['nucleofug_Boltz', 'Cone_angle_Emin', 'LUMO_Emin', 'dipole_Emin', 'Cone_angle_Boltz', 'Pyr_P_angle_Boltz', 'Sterimol_bur_B_5_Boltz', 'HOMO_Boltz', 'Sterimol_bur_B_5_Emin', 'HOMO_Emin', 'EA_Emin']
    # feature_var = [x + '_ligand' for x in feature_var]
    feature_var = ['Solid_angle_G_Emin_ligand', 'nucleofug_Boltz_ligand', 'IP_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'nucleo_Emin_react', 'Dispersion_area_Boltz_ligand', 'Pyr_P_Boltz_react', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'VBur_Emin_ligand', 'Sterimol_bur_B_5_Emin_react', 'Dispersion_area_Emin_ligand', 'Solid_angle_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_Emin_ligand', 'Pyr_alpha_Emin_react', 'Solid_angle_G_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'HOMO_Boltz_react']
    #feature_var = ['Solid_angle_G_Emin_ligand', 'VBur_Emin_react', 'electrofug_Emin_ligand', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'Dispersion_vol_Emin_react', 'VBur_Emin_ligand', 'Pyr_alpha_Emin_react', 'Sterimol_bur_B_1_Emin_ligand', 'Dispersion_vol_Boltz_react', 'Sterimol_bur_L_Boltz_ligand', 'Dispersion_area_Boltz_react', 'Sterimol_bur_L_Emin_ligand']
    # feature_var = ['nucleofug_Boltz_ligand', 'Pyr_P_Emin_react', 'Sterimol_B_1_Boltz_ligand', 'Sterimol_bur_L_Emin_react', 'Pyr_alpha_Emin_react', 'Sterimol_bur_B_5_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'Sterimol_bur_B_5_Boltz_react', 'Sterimol_bur_L_Emin_ligand', 'Sterimol_B_1_Emin_ligand', 'Pyr_alpha_Boltz_react', 'Sterimol_bur_B_1_Emin_ligand', 'Pyr_P_angle_Boltz_ligand', 'Sterimol_L_Boltz_ligand', 'Buried_mol_VBur_Emin_react', 'Sterimol_bur_L_Boltz_ligand', 'Pyr_P_angle_Emin_ligand', 'Sterimol_L_Emin_ligand']

    # m = GaussianProcessModel(components=[], length_scale_bounds=(1e-4, 1e3), restrict=True,
    #                          kernel=1**2 * RBF(length_scale=1) + WhiteKernel(noise_level=1))

    m = SVRModel(components=[MinMaxScaler()])
    #m = LogRegModel(components=[MinMaxScaler()])
    #m = RFModel(components=[OrdinalEncoder(), MinMaxScaler()], restrict=True)
    m.assign_params([3.        , 0.61906065, 0.        , 1.        , 0.1       ])

    metrics = Metrics(plot_res=False)
    logging.info('Running model evaluation given a feature set')
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    compounds = [''] * len(df)
    res, preds = m.eval_loo(df, obj_var, feature_var, metrics, compounds, print_res=True, plot_res=plot_res)

    error_plot(df[obj_var[0]], preds, 'SVR')
    print(f'R2 score: {r2_score(df[obj_var[0]], preds)}')
    print(preds)


def var_sel(df_path):
    df = pd.read_csv(df_path)
    df.dropna(axis='rows', inplace=True)
    df.reset_index(inplace=True, drop=True)
    # rdkit_lig = extract_rdkit_dataset(df['smiles_react'].values)
    # rdkit_lig.columns = [x + '_react' for x in rdkit_lig.columns]
    # df = pd.concat([df, rdkit_lig], axis=1)

    df = df.select_dtypes(exclude=['object'])  # Remove columns with strings
    # Add white noise to the 98:2 and 99:1 extremes
    df.loc[df['ratio'] < -2.2, 'ratio'] = df.loc[df['ratio'] < -2.2, 'ratio'] + np.random.normal(0, 0.1, len(df.loc[df['ratio'] < -2.2, 'ratio']))

    #subset = []

    obj_var = ["ratio"]

    #df = df[list(set(subset))+obj_var]

    ini_feat = []
    # ini_feat = ['Solid_angle_G_Emin_ligand', 'nucleofug_Boltz_ligand', 'IP_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'nucleo_Emin_react', 'Dispersion_area_Boltz_ligand', 'Pyr_P_Boltz_react', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'VBur_Emin_ligand', 'Sterimol_bur_B_5_Emin_react', 'Dispersion_area_Emin_ligand', 'Solid_angle_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_Emin_ligand', 'Pyr_alpha_Emin_react', 'Solid_angle_G_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand']
    components = [MinMaxScaler()]

    metrics = Metrics(default_cat='1_minus_f1')
    sel = GreedyVarSel(RidgeModel, metrics, components=components, max_features=40, force=True)
    #sel = GreedyVarSel(SVRModel, metrics, components=components, max_features=20, force=True)
    #sel = BackwardsGreedyVarSel(LinRegModel, metrics, components=components, min_features=5)
    #sel = GeneticVarSel(SVRModel, metrics, components=components, max_it=300, n_inds=15, per_mut=0.3, max_features=20)

    logging.info('Running feature subset selection')
    logging.info(sel.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Initial feature set:\n' + str(ini_feat))

    sel.run(df, obj_var, ini_feat=ini_feat, folds=5, shuffle=True, seed=42)  #folds=len(df))


def optim_model(df_path):
    df = pd.read_csv(df_path)
    # df.dropna(axis='rows', inplace=True)
    # df = df.select_dtypes(exclude=['object'])  # Remove columns with strings

    obj_var = ["ratio"]
    feature_var = ['Solid_angle_G_Emin_ligand', 'nucleofug_Boltz_ligand', 'IP_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'nucleo_Emin_react', 'Dispersion_area_Boltz_ligand', 'Pyr_P_Boltz_react', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'VBur_Emin_ligand', 'Sterimol_bur_B_5_Emin_react', 'Dispersion_area_Emin_ligand', 'Solid_angle_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_Emin_ligand', 'Pyr_alpha_Emin_react', 'Solid_angle_G_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand']

    metrics = Metrics(plot_res=False, default_cat='1_minus_f1')
    model = SVRModel()
    bounds = [(2.0, 3.5), (1e-2, 5.0), (0.0, 5.0), (1e-2, 5.0), (1e-2, 5.0)]  # degree, gamma, coef0, C, epsilon for SVR
    # bounds = [(1e-1, 1e1), (0.0, 1.0)]  # alpha, l1 for ElasticNet
    popsize = 5
    init = [[2.0, 0.01, 0.0, 1.0, 0.1], [3.0, 0.1, 0.0, 1.0, 0.1], [3.0, 1.0, 0.0, 1.0, 0.1], [3.0, 2.0, 0.0, 1.0, 0.1]]
    # init = [[1.0, 0.5]]  # Initial population
    optim = DiffEvOptim(model, metrics, bounds, maxiter=50, popsize=popsize)

    logging.info('Running hyperparameter optimization')
    logging.info(optim.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    res = optim.run(df, obj_var, feature_var, init=init, folds=len(df),
                    shuffle=True, seed=42, splits=None, print_res=True)

    logging.info(f'Best parameters found: {res}')


def getMolDescriptors(smiles, missingVal=None):
    """Calculate all RDKit descriptors for a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        res[nm] = val
    return res


def extract_rdkit_dataset(smiles):
    res = []
    cols = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA14', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SlogP_VSA1', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA5', 'SlogP_VSA6', 'TPSA', 'EState_VSA1', 'EState_VSA2', 'EState_VSA4', 'EState_VSA5', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'FractionCSP3', 'HeavyAtomCount', 'NOCount', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_benzene']
    for s in smiles:
        d = getMolDescriptors(s)
        vals = [d[key] for key in cols]
        res.append(vals)

    return pd.DataFrame(res, columns=cols)


def loo_and_validate(df_path):
    df = pd.read_csv(df_path)
    df.dropna(axis='rows', inplace=True)
    # df = df.sample(frac=1, random_state=42)  # Shuffle the dataset
    # Add white noise to the 98:2 and 99:1 extremes
    df.loc[df['ratio'] < -2.2, 'ratio'] = df.loc[df['ratio'] < -2.2, 'ratio'] + np.random.normal(0, 0.1, len(df.loc[df['ratio'] < -2.2, 'ratio']))

    df_val = df.sample(frac=0.1, random_state=43)
    df = df[[x not in df_val.index for x in df.index]]

    obj_var = ["ratio"]

    plot_res = False

    # SVR
    feature_var = ['Solid_angle_G_Emin_ligand', 'nucleofug_Boltz_ligand', 'IP_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'nucleo_Emin_react', 'Dispersion_area_Boltz_ligand', 'Pyr_P_Boltz_react', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'VBur_Emin_ligand', 'Sterimol_bur_B_5_Emin_react', 'Dispersion_area_Emin_ligand', 'Solid_angle_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_Emin_ligand', 'Pyr_alpha_Emin_react', 'Solid_angle_G_Boltz_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'HOMO_Boltz_react']
    # GP
    feature_var = ['Solid_angle_G_Emin_ligand', 'VBur_Emin_react', 'electrofug_Emin_ligand', 'Sterimol_bur_B_5_Boltz_react', 'Pyr_alpha_Boltz_react', 'Sterimol_B_1_Boltz_ligand', 'Dispersion_vol_Emin_react', 'VBur_Emin_ligand', 'Pyr_alpha_Emin_react', 'Sterimol_bur_B_1_Emin_ligand', 'Dispersion_vol_Boltz_react', 'Sterimol_bur_L_Boltz_ligand', 'Dispersion_area_Boltz_react', 'Sterimol_bur_L_Emin_ligand']
    # RF
    # feature_var = ['Sterimol_B_1_Boltz_ligand', 'Sterimol_bur_L_Emin_react', 'Sterimol_bur_L_Boltz_react', 'Sterimol_bur_B_1_Boltz_ligand', 'Sterimol_L_Emin_ligand', 'nucleofug_Emin_react', 'nucleo_Boltz_ligand', 'Sasa_mol_VBur_Boltz_react', 'EA_Boltz_react', 'Solid_angle_G_Boltz_ligand', 'electro_Emin_react', 'Dispersion_vol_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'Sasa_mol_VBur_Emin_ligand', 'SASA_area_Emin_react', 'Sterimol_B_1_Emin_ligand', 'Sterimol_bur_B_5_Emin_ligand', 'dipole_Boltz_ligand', 'Free_VBur_Boltz_react', 'nucleo_Emin_ligand', 'SASA_area_Boltz_react', 'Sasa_distal_VBur_Emin_react']
    # Ridge
    # feature_var = ['Free_VBur_Emin_ligand', 'LUMO_Emin_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'Sasa_mol_VBur_Emin_react', 'Sterimol_L_Emin_ligand', 'Free_VBur_Boltz_react', 'electro_Boltz_ligand', 'VBur_Emin_ligand', 'Pyr_P_angle_Emin_react', 'Sterimol_B_5_Boltz_ligand', 'EA_Boltz_ligand', 'Sterimol_L_Boltz_ligand', 'SASA_area_Boltz_ligand', 'Sterimol_B_5_Boltz_react', 'Sterimol_B_1_Emin_react', 'electro_Emin_ligand', 'nucleofug_Boltz_react', 'VBur_Boltz_react', 'Solid_angle_G_Emin_ligand', 'SASA_area_Emin_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'Dispersion_P_int_Boltz_ligand', 'Cone_angle_NAtoms_Emin_ligand', 'Dispersion_P_int_Emin_ligand', 'Pyr_alpha_Emin_ligand', 'VBur_Boltz_ligand', 'Sterimol_bur_B_5_Emin_ligand', 'electrofug_Emin_ligand', 'Sasa_mol_VBur_Emin_ligand', 'Pyr_P_angle_Boltz_ligand', 'Sasa_mol_VBur_Boltz_ligand', 'Buried_mol_VBur_Emin_ligand', 'electrofug_Boltz_ligand', 'Buried_mol_VBur_Boltz_ligand', 'Dispersion_area_Boltz_ligand', 'Sasa_distal_VBur_Boltz_ligand', 'Pyr_P_angle_Emin_ligand', 'Buried_distal_VBur_Boltz_ligand', 'Dispersion_area_Emin_ligand', 'Sasa_distal_VBur_Emin_ligand']

    m = GaussianProcessModel(components=[MinMaxScaler()])
    #m.assign_params([3., 0.61906065, 0., 1., 0.1])

    metrics = Metrics(plot_res=False)
    logging.info('Running model evaluation given a feature set')
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    compounds = [''] * len(df)
    res, preds = m.eval_loo(df, obj_var, feature_var, metrics, compounds, print_res=True, plot_res=plot_res)

    error_plot(df[obj_var[0]], preds, 'Gaussian process')
    print(f'R2 score: {r2_score(df[obj_var[0]], preds)}')
    print(preds)

    # Validation
    m = SVRModel(components=[MinMaxScaler()])
    m.assign_params([3., 0.61906065, 0., 1., 0.1])
    m.fit(df, obj_var, feature_var, split=False, print_res=False)

    res = m.predict(df_val)
    error_plot(df_val[obj_var[0]], res, 'Gaussian process')
    print(f'R2 score: {r2_score(df_val[obj_var[0]], res)}')



if __name__ == '__main__':
    # Prepare the logger to output into both console and a file with the desired format
    date = datetime.now()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    #file_log_handler = logging.FileHandler(filename=date.strftime('reactions_%d_%m_%Y_%H_%M.log'), mode='w')
    #logger.addHandler(file_log_handler)

    logging.info('Started')
    #eval_model(sys.argv[1])

    #plot_aggregate_results(sys.argv[1])
    # eval_loo_model(sys.argv[1])
    loo_and_validate(sys.argv[1])
    # eval_model(sys.argv[1])
    # optim_model(sys.argv[1])
    # var_sel(sys.argv[1])
    # train_and_save(sys.argv[1])

    logging.info('Finished')
