import numpy as np


def trajLoss(target_traj, traj, diff_traj=None, x_w=None, u_w=None):
    # target trajectory
    target_u_traj = target_traj['u_traj']
    target_x_traj = target_traj['x_traj']

    # variable trajectory
    u_traj = traj['u_opt_traj']
    x_traj = traj['x_opt_traj']

    if x_w is not None:
        x_w = np.diag(x_w)
    else:
        x_w = np.diag(1. / ((target_x_traj * target_x_traj).mean(axis=0)+0.0001))

    if u_w is not None:
        u_w = np.diag(u_w)
    else:
        u_w = np.diag(1. / ((target_u_traj * target_u_traj).mean(axis=0)+0.0001))

    if diff_traj is not None:
        # gradient of trajectory
        jac_x2aux_traj = diff_traj['jac_x2aux_traj']
        jac_u2aux_traj = diff_traj['jac_u2aux_traj']

    # compute the loss
    horizon = u_traj.shape[0]
    loss = 0.0

    if diff_traj is not None:
        loss_grad = np.zeros(jac_x2aux_traj.shape[-1])

    for t in range(horizon):

        # ----------debug: another choice of  weights (this seems not good)
        # u_w = np.diag(1. / (target_u_traj[t] * target_u_traj[t] + 0.00001))
        # x_w = np.diag(1. / (target_x_traj[t] * target_x_traj[t] + 0.00001))

        # compute the loss
        loss += np.dot(target_u_traj[t] - u_traj[t], u_w @ (target_u_traj[t] - u_traj[t]))
        loss += np.dot(target_x_traj[t] - x_traj[t], x_w @ (target_x_traj[t] - x_traj[t]))

        if diff_traj is not None:
            # compute the gradient
            loss_grad += u_w @ (u_traj[t] - target_u_traj[t]) @ jac_u2aux_traj[t]
            loss_grad += x_w @ (x_traj[t] - target_x_traj[t]) @ jac_x2aux_traj[t]

    # ----------debug: another choice of  weights
    # x_w = np.diag(1. / (0.00001 + target_x_traj[-1] * target_x_traj[-1]))

    # the final loss
    loss += np.dot(target_x_traj[-1] - x_traj[-1], x_w @ (target_x_traj[-1] - x_traj[-1]))

    if diff_traj is not None:
        loss_grad += x_w @ (x_traj[-1] - target_x_traj[-1]) @ jac_x2aux_traj[-1]
        loss_grad = loss_grad.flatten()
    else:
        loss_grad = None

    return loss, loss_grad
