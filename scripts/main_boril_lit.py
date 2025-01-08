import sys
import logging
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from mlworkflow.models.regression_models import LinRegModel, RidgeModel, RFModel, SVRModel, ElasticModel, LassoModel, GaussianProcessModel
from mlworkflow.models.greedy_var_sel import GreedyVarSel, BackwardsGreedyVarSel
from mlworkflow.models.genetic_var_sel import GeneticVarSel
from mlworkflow.models.metrics import Metrics
from mlworkflow.models.diff_ev_optim import DiffEvOptim
from mlworkflow.utils import find_k_tani
import pickle


def remove_str_cols(df):
    """ Remove the columns that have strings in their first row in the dataframe """
    idx = [not isinstance(df[x][0], str) for x in df.columns]

    return df.loc[:, idx]


def train_and_save(output_path):
    df = pd.read_csv(output_path)
    df.dropna(axis='rows', inplace=True)
    df = df.select_dtypes(exclude=['object'])  # Remove columns with strings

    obj_var = ["yield"]

    feature_var = ['Sterimol_B_5_Boltz_ligand', 'E_thermal_correction_react_1', 'Sasa_mol_VBur_Emin_ligand', 'IP_Boltz_react_1', 'Sasa_distal_VBur_Boltz_ligand']
    #params = [3.04303804, 0.42729992, 0.19825187, 3.37142002, 0.0950036]

    logging.info('Training and saving model with all data')
    m = SVRModel(kernel='poly')
    #m.assign_params(params)
    m.fit(df, obj_var, feature_var, split=False, print_res=False)
    m.save_model()
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))


def eval_model(output_path):
    df = pd.read_csv(output_path)
    df = remove_str_cols(df)

    obj_var = ["yield"]

    feature_var = ['Sterimol_B_5_Boltz_ligand', 'E_thermal_correction_react_1', 'Sasa_mol_VBur_Emin_ligand', 'IP_Boltz_react_1', 'Sasa_distal_VBur_Boltz_ligand']

    m = SVRModel()
    metrics = Metrics(plot_res=False)
    #m.assign_params([3.02903645e+00, 3.10060228e+00, 3.07712732e-04, 1.04555851e+00, 6.50537473e-02])
    logging.info('Running model evaluation given a feature set')
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    # Splits ignoring the IPr on position 2
    kf = KFold(n_splits=15)
    splits = kf.split(df)
    splits = [(x, y) for (x, y) in splits]
    splits.pop(2)

    res = m.eval_cv(df, obj_var, feature_var, metrics, splits=splits, print_res=True)

    logging.info(f'Average error obtained: {res}')


def eval_loo_model(output_path):
    df = pd.read_csv(output_path)
    df = df.select_dtypes(exclude=['object'])  # Remove columns with strings
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 52, 53, 55, 56, 57, 62, 68, 69, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 108, 109]
    idx = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 58, 59, 60, 61, 63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 76, 84, 91, 97, 107]
    #df = df.iloc[idx]
    # exclude_df = df.index.isin(range(51, 58))
    # df = df[~exclude_df]
    #df = df.sample(frac=1)
    #df.reset_index(inplace=True)
    compounds = [''] * len(df)

    plot_res = True

    obj_var = ["ratio"]

    # LinReg 0.6824
    feature_var = ['Cone_angle_Emin_ligand', 'HOMO_Boltz_ligand', 'Sterimol_bur_B_5_Emin_ligand', 'EA_Emin_ligand', 'G_ligand', 'dipole_Emin_ligand', 'Dispersion_area_Emin_ligand', 'electronic_spatial_extent_ligand', 'HOMO_Emin_react', 'Cone_angle_NAtoms_Boltz_ligand', 'hardness_ligand']

    # Ridge 0.62
    #feature_var = ['Cone_angle_Emin_ligand', 'dipole_Boltz_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'nucleofug_Emin_ligand', 'electronegativity_ligand', 'HOMO_Emin_react', 'Sterimol_bur_B_5_Emin_ligand', 'E_scf_ligand', 'Sterimol_B_5_Emin_ligand', 'Cone_angle_NAtoms_Emin_ligand', 'nucleofug_Boltz_ligand', 'SASA_area_Emin_ligand', 'LUMO_Boltz_ligand', 'dipole_Emin_ligand', 'Sterimol_B_5_Boltz_ligand', 'E_zpe_ligand', 'HOMO_Emin_ligand', 'EA_Emin_ligand', 'LUMO_Emin_ligand', 'SASA_vol_Boltz_ligand']

    # SVR 0.52
    #feature_var = ['number_of_atoms_ligand', 'charge_ligand', 'dipole_ligand', 'electronic_spatial_extent_ligand', 'G_ligand', 'homo_energy_ligand', 'electronegativity_ligand', 'hardness_ligand', 'IP_Emin_ligand', 'EA_Boltz_ligand', 'HOMO_Emin_ligand', 'dipole_Boltz_ligand', 'nucleo_Boltz_ligand', 'nucleo_Emin_ligand', 'electrofug_Boltz_ligand', 'SASA_area_Boltz_ligand', 'SASA_area_Emin_ligand', 'Dispersion_area_Boltz_ligand', 'Dispersion_area_Emin_ligand', 'Dispersion_vol_Emin_ligand', 'Dispersion_P_int_Emin_ligand', 'Free_VBur_Boltz_ligand', 'Sasa_mol_VBur_Emin_ligand', 'Buried_mol_VBur_Boltz_ligand', 'Pyr_alpha_Emin_ligand', 'Cone_angle_Boltz_ligand', 'Cone_angle_Emin_ligand', 'Solid_angle_Emin_ligand', 'Solid_angle_G_Boltz_ligand', 'Cone_angle_NAtoms_Emin_ligand', 'Sterimol_L_Boltz_ligand', 'Sterimol_B_1_Emin_ligand', 'Sterimol_B_5_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'Sterimol_bur_L_Emin_ligand', 'Sterimol_bur_B_1_Boltz_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'IP_Emin_react', 'EA_Boltz_react', 'EA_Emin_react', 'HOMO_Boltz_react', 'dipole_Boltz_react', 'dipole_Emin_react', 'electro_Boltz_react', 'electro_Emin_react', 'nucleofug_Boltz_react', 'nucleofug_Emin_react', 'SASA_area_Boltz_react', 'SASA_area_Emin_react', 'Dispersion_vol_Boltz_react', 'Dispersion_P_int_Boltz_react', 'Dispersion_P_int_Emin_react']
    
    # RF 0.42
    #feature_var = ['charge_ligand', 'dipole_ligand', 'H_thermal_correction_ligand', 'E_zpe_ligand', 'E_ligand', 'G_ligand', 'lumo_energy_ligand', 'electronegativity_ligand', 'hardness_ligand', 'EA_Boltz_ligand', 'EA_Emin_ligand', 'LUMO_Emin_ligand', 'electro_Emin_ligand', 'electrofug_Boltz_ligand', 'nucleofug_Boltz_ligand', 'SASA_area_Boltz_ligand', 'SASA_area_Emin_ligand', 'SASA_vol_Emin_ligand', 'Dispersion_P_int_Boltz_ligand', 'VBur_Boltz_ligand', 'VBur_Emin_ligand', 'Free_VBur_Boltz_ligand', 'Free_VBur_Emin_ligand', 'Sasa_distal_VBur_Boltz_ligand', 'Sasa_mol_VBur_Emin_ligand', 'Buried_mol_VBur_Boltz_ligand', 'Pyr_alpha_Emin_ligand', 'Pyr_P_angle_Boltz_ligand', 'Cone_angle_Boltz_ligand', 'Cone_angle_Emin_ligand', 'Solid_angle_Emin_ligand', 'Solid_angle_G_Boltz_ligand', 'Cone_angle_NAtoms_Boltz_ligand', 'Sterimol_L_Boltz_ligand', 'Sterimol_L_Emin_ligand', 'Sterimol_B_1_Emin_ligand', 'Sterimol_B_5_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'Sterimol_bur_B_1_Emin_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'IP_Boltz_react', 'IP_Emin_react', 'EA_Boltz_react', 'EA_Emin_react', 'HOMO_Emin_react', 'LUMO_Boltz_react', 'dipole_Boltz_react', 'dipole_Emin_react', 'nucleo_Boltz_react', 'nucleo_Emin_react', 'electrofug_Boltz_react', 'electrofug_Emin_react', 'nucleofug_Emin_react', 'SASA_vol_Boltz_react', 'SASA_vol_Emin_react', 'Dispersion_vol_Boltz_react', 'Dispersion_P_int_Boltz_react']
    feature_var = ['number_of_atoms_ligand', 'multiplicity_ligand', 'electronegativity_ligand', 'hardness_ligand', 'IP_Boltz_ligand', 'IP_Emin_ligand', 'EA_Boltz_ligand', 'LUMO_Boltz_ligand', 'dipole_Emin_ligand', 'electro_Boltz_ligand', 'electro_Emin_ligand', 'nucleo_Boltz_ligand', 'electrofug_Boltz_ligand', 'SASA_area_Emin_ligand', 'SASA_vol_Emin_ligand', 'Dispersion_vol_Emin_ligand', 'Sasa_mol_VBur_Emin_ligand', 'Buried_distal_VBur_Emin_ligand', 'Pyr_alpha_Boltz_ligand', 'Pyr_P_Boltz_ligand', 'Pyr_P_angle_Emin_ligand', 'Cone_angle_Emin_ligand', 'Solid_angle_G_Boltz_ligand', 'Solid_angle_G_Emin_ligand', 'Cone_angle_NAtoms_Boltz_ligand', 'Cone_angle_NAtoms_Emin_ligand', 'Sterimol_B_5_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'Sterimol_bur_L_Emin_ligand', 'Sterimol_bur_B_1_Emin_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'IP_Boltz_react', 'HOMO_Emin_react', 'dipole_Emin_react', 'electro_Emin_react', 'nucleo_Boltz_react', 'electrofug_Boltz_react', 'nucleofug_Emin_react', 'SASA_area_Boltz_react', 'Dispersion_area_Emin_react', 'Dispersion_vol_Boltz_react']

    #m = GaussianProcessModel(components=[MinMaxScaler()], length_scale_bounds=(1e-5, 1e5), restrict=True)

    m = RFModel(components=[MinMaxScaler()])
    #m.assign_params([10, 4, 10, 4])
    m.assign_params([30, 5, 10, 4])

    metrics = Metrics(plot_res=False)
    logging.info('Running model evaluation given a feature set')
    logging.info(m.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    res, preds = m.eval_loo(df, obj_var, feature_var, metrics, compounds, print_res=True, plot_res=plot_res)
    print(f'R2 score: {r2_score(df["ratio"], preds)}')

    # import numpy as np
    # import matplotlib.pyplot as plt
    # importances = m.pipe[1].feature_importances_
    # std = np.std([tree.feature_importances_ for tree in m.pipe[1].estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_var)
    #idx = [2, 7, 11, 13, 22, 26, 30, 33, 39, 44, 47, 51, 56]

    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # #forest_importances[idx].plot.bar(yerr=std[idx], ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    # from sklearn.inspection import permutation_importance
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(df[feature_var].values, df[obj_var].values)
    # result = permutation_importance(m.pipe[1], X_test, y_test, n_repeats=50, n_jobs=2)
    # forest_importances = pd.Series(result.importances_mean, index=feature_var)
    #
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()


def var_sel(output_path):
    df = pd.read_csv(output_path)
    df = df.select_dtypes(exclude=['object'])  # Remove columns with strings

    #subset = []
    #subset = ['Cone_angle_NAtoms_Boltz_ligand', 'LUMO_Emin_ligand', 'dipole_react_1', 'IP_Emin_ligand', 'charge_react_1', 'Pyr_alpha_Boltz_ligand', 'Sterimol_B_5_Boltz_ligand', 'SASA_area_Emin_react_1', 'IP_Boltz_react_1', 'dipole_Boltz_ligand', 'multiplicity_react_1', 'Pyr_alpha_Emin_ligand', 'hardness_react_1', 'Sterimol_bur_L_Boltz_ligand', 'dipole_Boltz_react_1', 'IP_Boltz_ligand', 'Dispersion_vol_Boltz_ligand', 'Sterimol_B_1_Boltz_ligand', 'Sterimol_L_Emin_ligand', 'Sterimol_L_Boltz_ligand', 'Cone_angle_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'homo_energy_react_1', 'Pyr_P_angle_Emin_ligand', 'HOMO_Boltz_react_1', 'Dispersion_P_int_Emin_ligand', 'dipole_Emin_ligand', 'Sterimol_B_1_Emin_ligand', 'Solid_angle_G_Boltz_ligand', 'Cone_angle_Emin_ligand', 'HOMO_Emin_react_1', 'Buried_distal_VBur_Emin_ligand', 'Pyr_P_Boltz_ligand', 'electronic_spatial_extent_react_1', 'Dispersion_vol_Emin_ligand']
    obj_var = ["ratio"]

    y = df[obj_var]
    #df = df[list(set(subset))+obj_var]

    ini_feat = []
    #ini_feat = ['Pyr_P_Boltz_ligand', 'hardness_react_1', 'Sterimol_L_Boltz_ligand', 'dipole_Boltz_ligand', 'homo_energy_react_1', 'dipole_react_1', 'IP_Boltz_react_1', 'IP_Emin_ligand', 'Dispersion_P_int_Emin_ligand', 'IP_Boltz_ligand', 'dipole_Emin_ligand']
    components = [MinMaxScaler()]

    metrics = Metrics()
    #sel = GreedyVarSel(RFModel, metrics, components=components, max_features=20, force=True)
    sel = GreedyVarSel(LinRegModel, metrics, components=components, max_features=20, force=True)
    #sel = BackwardsGreedyVarSel(LinRegModel, metrics, components=components, min_features=5)
    #sel = GeneticVarSel(RFModel, metrics, components=components, max_it=300, n_inds=15, per_mut=0.3, max_features=20)

    logging.info('Running feature subset selection')
    logging.info(sel.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Initial feature set:\n' + str(ini_feat))

    sel.run(df, obj_var, ini_feat=ini_feat, folds=len(y))#, shuffle=True, seed=42)


def optim_model(output_path):
    df = pd.read_csv(output_path)
    df = df.select_dtypes(exclude=['object'])  # Remove columns with strings

    obj_var = ['ratio']
    feature_var = ['number_of_atoms_ligand', 'multiplicity_ligand', 'electronegativity_ligand', 'hardness_ligand', 'IP_Boltz_ligand', 'IP_Emin_ligand', 'EA_Boltz_ligand', 'LUMO_Boltz_ligand', 'dipole_Emin_ligand', 'electro_Boltz_ligand', 'electro_Emin_ligand', 'nucleo_Boltz_ligand', 'electrofug_Boltz_ligand', 'SASA_area_Emin_ligand', 'SASA_vol_Emin_ligand', 'Dispersion_vol_Emin_ligand', 'Sasa_mol_VBur_Emin_ligand', 'Buried_distal_VBur_Emin_ligand', 'Pyr_alpha_Boltz_ligand', 'Pyr_P_Boltz_ligand', 'Pyr_P_angle_Emin_ligand', 'Cone_angle_Emin_ligand', 'Solid_angle_G_Boltz_ligand', 'Solid_angle_G_Emin_ligand', 'Cone_angle_NAtoms_Boltz_ligand', 'Cone_angle_NAtoms_Emin_ligand', 'Sterimol_B_5_Boltz_ligand', 'Sterimol_B_5_Emin_ligand', 'Sterimol_bur_L_Emin_ligand', 'Sterimol_bur_B_1_Emin_ligand', 'Sterimol_bur_B_5_Boltz_ligand', 'IP_Boltz_react', 'HOMO_Emin_react', 'dipole_Emin_react', 'electro_Emin_react', 'nucleo_Boltz_react', 'electrofug_Boltz_react', 'nucleofug_Emin_react', 'SASA_area_Boltz_react', 'Dispersion_area_Emin_react', 'Dispersion_vol_Boltz_react']

    metrics = Metrics(plot_res=False)
    model = RFModel()
    bounds = [(10.0, 30.0), (2.0, 5.0), (5.0, 15.0), (3.0, 10.0)]  # [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    popsize = 10
    init = [[15.0, 3.0, 5.0, 3.0], [30.0, 5.0, 10.0, 5.0], [20.0, 3.0, 5.0, 3.0], [10.0, 2.0, 5.0, 3.0]]
    optim = DiffEvOptim(model, metrics, bounds, maxiter=100, popsize=popsize)

    logging.info('Running hyperparameter optimization')
    logging.info(optim.to_str())
    logging.info('Objective variable: ' + str(obj_var))
    logging.info('Feature set:\n' + str(feature_var))

    res = optim.run(df, obj_var, feature_var, folds=len(df), init=init,
                    shuffle=False, seed=None, splits=None, print_res=True)

    logging.info(f'Best parameters found: {res}')


if __name__ == '__main__':
    # Prepare the logger to output into both console and a file with the desired format
    date = datetime.now()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    #file_log_handler = logging.FileHandler(filename=date.strftime('reactions_%d_%m_%Y_%H_%M.log'), mode='w')
    #logger.addHandler(file_log_handler)

    logging.info('Started')
    #eval_model(sys.argv[1])

    eval_loo_model(sys.argv[1])
    #optim_model(sys.argv[1])
    #var_sel(sys.argv[1])
    #train_and_save(sys.argv[1])

    logging.info('Finished')
