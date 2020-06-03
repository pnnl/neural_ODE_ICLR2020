"""
Neural State Space Models - N-step ahead System ID
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import sys
sys.path.append(os.path.abspath('../'))
os.chdir('../')
from system_id_nlin_con import make_dataset_con, SSM_black_con, SSM_gray_con, SSM_white_con, Building_DAE


def plot_trajectories(traj1, traj2, traj3, traj4, labels, figname):
    print(len(traj1))
    fig, ax = plt.subplots(len(traj1), 1, gridspec_kw={'wspace':0, 'hspace':0})#, constrained_layout=True)
    for row, (t1, t2, t3, t4, label) in enumerate(zip(traj1, traj2, traj3, traj4, labels)):
        ax[row].plot(t1, label=f'True')
        ax[row].plot(t2, '--', label='cODE$_B$')
        ax[row].plot(t3, '-.', label='cODE$_G$')
        ax[row].plot(t4, ':', label='cODE$_W$')
        steps = range(0, t1.shape[0] + 1, 288)
        days = np.array(list(range(len(steps))))+7
        ax[row].set(xticks=steps,
                    xticklabels=days,
                    xlim=(0, len(t1)))
        ax[row].set_ylabel(label)
        ax[row].tick_params(labelbottom=False)
        ax[row].axvspan(2016, 4032, facecolor='grey', alpha=0.25, zorder=-100)
        ax[row].axvspan(4032, 6048, facecolor='grey', alpha=0.5, zorder=-100)
    plt.text(1500, 14, "Train", fontsize=8)
    plt.text(3200, 14, "Validation", fontsize=8)
    plt.text(5600, 14, "Test", fontsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)

    ax[-1].tick_params(labelbottom=True)
    plt.legend(fontsize=7, labelspacing=0.2, ncol=2, loc='upper left', facecolor='white')
    plt.tight_layout()
    plt.savefig(f'{figname}.pdf')
    plt.savefig(f'{figname}.png')

####################################
###### DATA SETUP
####################################
train_data, dev_data, test_data = make_dataset_con(16, 'cpu')

ny, nx, n_m, n_dT, nu, nd, n_hidden = 1, 4, 1, 1, 1, 3, 8

ssmwhite = SSM_white_con(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
ssmwhite.load_state_dict(torch.load('iclr_models/ssmwhitecon128.pth', map_location=torch.device('cpu')))

ssmgray = SSM_gray_con(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
ssmgray.load_state_dict(torch.load('iclr_models/ssmgraycon128.pth', map_location=torch.device('cpu')))

ssmblack = SSM_black_con(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
ssmblack.load_state_dict(torch.load('iclr_models/ssmblackcon128.pth', map_location=torch.device('cpu')))

building = Building_DAE()
criterion = nn.MSELoss()  # we'll convert this to RMSE later

#######################################
######### OPEN LOOP RESPONSE ####
#######################################
def open_loop(model, data):
    data = torch.cat([data[:, k, :] for k in range(data.shape[1])]).unsqueeze(1)
    x0_in, M_flow_in, DT_in, D_in, XMIN_in, XMAX_in, UMIN_in, UMAX_in, x_response, Y_target, Sx_targets, Su_targets = (
    data[0, :, :nx],
    data[:, :, nx:nx + n_m],
    data[:, :, nx + n_m:nx + n_m + n_dT],
    data[:, :, nx + n_m + n_dT:nx + n_m + n_dT + nd],
    data[:, :, nx + n_m + n_dT + nd:nx + n_m + n_dT + nd + nx],
    data[:, :, nx + n_m + n_dT + nd + nx:nx + n_m + n_dT + nd + nx + nx],
    data[:, :, nx + n_m + n_dT + nd + nx + nx:nx + n_m + n_dT + nd + nx + nx + nu],
    data[:, :, nx + n_m + n_dT + nd + nx + nx + nu:nx + n_m + n_dT + nd + nx + nx + nu + nu],
    data[:, :, nx + n_m + n_dT + nd + nx + nx + nu + nu:nx + n_m + n_dT + nd + nx + nx + nu + nu + nx],
    data[:, :, nx + n_m + n_dT + nd + nx + nx + nu + nu + nx:nx + n_m + n_dT + nd + nx + nx + nu + nu + nx + ny],
    data[:, :,
    nx + n_m + n_dT + nd + nx + nx + nu + nu + nx + ny:nx + n_m + n_dT + nd + nx + nx + nu + nu + nx + ny + nx],
    data[:, :,
    nx + n_m + n_dT + nd + nx + nx + nu + nu + nx + ny + nx:nx + n_m + n_dT + nd + nx + nx + nu + nu + nx + ny + nx + nu])
    X_pred, Y_pred, Sx_mim_pred, Sx_max_pred, Su_mim_pred, Su_max_pred = model(x0_in, M_flow_in, DT_in, D_in,
                                                                               XMIN_in, XMAX_in, UMIN_in, UMAX_in)
    return (criterion(Y_pred.squeeze(), Y_target.squeeze()),
            X_pred.squeeze().detach().cpu().numpy(),
            x_response.squeeze().detach().cpu().numpy())


trajs = []
for m in [ssmblack, ssmgray, ssmwhite]:
    openloss, xpred, xtrue = open_loop(m, train_data)
    print(xtrue.shape)
    print(f' Train_open_loss: {openloss}')

    devopenloss, devxpred, devxtrue = open_loop(m, dev_data)
    print(f' Dev_open_loss: {devopenloss}')

    testopenloss, testxpred, testxtrue = open_loop(m, test_data)
    print(f' Test_open_loss: {testopenloss}')
    trajs.append(  [np.concatenate([xpred[:, k], devxpred[:, k], testxpred[:, k]])
                for k in range(xpred.shape[1])])


plot_trajectories([np.concatenate([xtrue[:, k], devxtrue[:, k], testxtrue[:, k]])
                   for k in range(xtrue.shape[1])],
             *trajs,
               ['$x_1$', '$x_2$', '$x_3$', '$x_4$'], 'con_open_test.png')


