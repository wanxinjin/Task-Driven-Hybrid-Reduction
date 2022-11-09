from casadi import *
import matplotlib.pyplot as plt

from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR

from util.logger import save_data, load_data
from diagnostics.lcs_analysis import LCSAnalyser

# ---------------------------- load the saved data ------------------------
save_dir = 'results'
saved_info = load_data(data_name='reduced_lcs_2d', save_dir=save_dir)

# ---------------------------- full model  --------------------------------
n_x, n_u, n_lam = saved_info['n_x'], saved_info['n_u'], saved_info['n_lam']
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam)
flcs_aux_val = saved_info['full_lcs_aux_val']
x0_mag = saved_info['x0_mag']

# ---------------------------- test mode count  ---------------------------
n_data = 100000
x_batch = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=(n_data, flcs.n_x))
u_batch = 10 * np.random.uniform(low=-1.0, high=1.0, size=(n_data, flcs.n_u))

# compute lam batch
lam_batch = []
for i in range(len(x_batch)):
    finfo = flcs.forwardDiff(aux_val=flcs_aux_val, x_val=x_batch[i],
                             u_val=u_batch[i], solver='qp')
    lam_batch.append(finfo['lam_val'])
lam_batch = np.array(lam_batch)

# object for lam analysis
analyser = LCSAnalyser()
flcs_stat = analyser.modeChecker(lam_batch)
flcs_unique_mode_id = flcs_stat['unique_mode_id']
flcs_n_unique_mode = flcs_stat['n_unique_mode']

print("full lcs mode count: ", flcs_n_unique_mode)
