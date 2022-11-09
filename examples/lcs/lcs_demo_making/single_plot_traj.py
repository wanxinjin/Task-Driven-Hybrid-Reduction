from casadi import *
import matplotlib.pyplot as plt

from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR

from env.util.rollout import rollout_mpcReceding_lcs, rollout_mpcOpenLoop_lcs

from util.logger import save_data, load_data
from diagnostics.vis_model import Visualizer
from diagnostics.lcs_analysis import LCSAnalyser

# ---------------------------- load the saved data ------------------------
save_dir = 'results'
saved_info = load_data(data_name='res_x20_lam15_rlam3', save_dir=save_dir)

# ---------------------------- full model  --------------------------------
n_x, n_u, n_lam = saved_info['n_x'], saved_info['n_u'], saved_info['n_lam']
flcs_stiff = saved_info['flcs_stiff']
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam, stiff=flcs_stiff)
flcs_aux_val = saved_info['flcs_aux_val']

#  --------------------------- reduced model ------------------------------
reduced_n_lam = saved_info['reduced_n_lam']
c = saved_info['c']
rlcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=reduced_n_lam, c=c)
rlcs_aux_val = saved_info['rlcs_aux_trace'][-1]

# ------------------ define the task cost function ------------------------
path_cost_fn = Function('path_cost_fn', [flcs.x, flcs.u], [dot(flcs.x, flcs.x) + dot(flcs.u, flcs.u)])
final_cost_fn = Function('final_cost_fn', [flcs.x], [dot(flcs.x, flcs.x)])

mpc_horizon = saved_info['mpc_horizon']
mpc_epsilon = saved_info['mpc_epsilon']
rollout_horizon = 8

# ---------------------- create a mpc for full lcs for comparison ---------
fmpc = MPCLCSR(lcs=flcs, epsilon=mpc_epsilon)
fmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
fmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- create a mpc for full lcs for comparison ---------
rmpc = MPCLCSR(lcs=rlcs, epsilon=mpc_epsilon)
rmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
rmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- compare -----------------------------------------
x0_mag = 2 * saved_info['x0_mag']
print(x0_mag)
vis = Visualizer()
analyser = LCSAnalyser()

n_rollout_mpc = 1
for i in range(n_rollout_mpc):
    x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)

    # ground truth mpc
    gt_mpc_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                             rollout_horizon=rollout_horizon,
                                             mpc=fmpc, mpc_aux=flcs_aux_val)
    gt_mpc_state_traj = gt_mpc_rollout['state_traj']
    gt_mpc_control_traj = gt_mpc_rollout['control_traj']
    gt_mpc_lam_traj = gt_mpc_rollout['lam_traj']
    print('ground truth mpc cost: ', gt_mpc_rollout['cost'])

    # learned model mpc
    mpc_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                          rollout_horizon=rollout_horizon,
                                          mpc=rmpc, mpc_aux=rlcs_aux_val)

    print('mpc cost: ', mpc_rollout['cost'])

    mpc_lam_traj = mpc_rollout['lam_traj']
    mpc_state_traj = mpc_rollout['state_traj']
    model_lam_traj = mpc_rollout['model_lam_traj']

    # do the animation
    state_lam_traj1 = dict(state_traj=gt_mpc_state_traj,
                           lam_traj=gt_mpc_lam_traj)
    state_lam_traj2 = dict(state_traj=mpc_state_traj,
                           lam_traj=model_lam_traj)
    analyser.plot_play4(state_lam_traj1=state_lam_traj1,
                        state_lam_traj2=state_lam_traj2, save=False)
