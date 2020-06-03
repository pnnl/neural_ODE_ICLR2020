"""
Neural State Space Models - N-step ahead System ID
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import scipy.linalg as LA

import os
import sys
sys.path.append(os.path.abspath('../'))
os.chdir('../')
from system_id_nlin import SSM_black, SSM_gray, SSM_white, RNN, Building_DAE
from system_id_nlin_con import SSM_black_con, SSM_gray_con, SSM_white_con, Building_DAE


def plot_matrices(matrices, labels, figname):
    rows = len(matrices)
    cols = len(matrices[0])
    fig, axes = plt.subplots(nrows=rows, ncols=cols)

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(matrices[i][j])
            axes[i, j].set(xticklabels= [],
                            xticks = [],
                            yticks=[],
                            yticklabels=[])
    axes[0, 0].set_xlabel('True System')
    axes[0, 1].set_xlabel('PI-RNN')
    axes[1, 0].set_xlabel('$ODE_B$')
    axes[1, 1].set_xlabel('$ODE_G$')
    axes[2, 0].set_xlabel('$ODE_w$')
    axes[2, 1].set_xlabel('$cODE_B$')
    axes[3, 0].set_xlabel('$cODE_G$')
    axes[3, 1].set_xlabel('$cODE_W$')
    plt.tight_layout()
    plt.savefig(figname+'.pdf')
    plt.savefig(figname+'.png')


nx, n_m, n_dT, nu, nd, n_hidden = 4, 1, 1, 1, 3, 8

ssmwhite = SSM_white(nx, n_m, n_dT, nu, nd, n_hidden, bias=False, device='cpu')
ssmwhite.load_state_dict(torch.load('iclr_models/ssmwhite128.pth', map_location=torch.device('cpu')))

ssmgray = SSM_gray(nx, n_m, n_dT, nu, nd, n_hidden, bias=False, device='cpu')
ssmgray.load_state_dict(torch.load('iclr_models/ssmgray64.pth', map_location=torch.device('cpu')))

ssmblack = SSM_black(nx, n_m, n_dT, nu, nd, n_hidden, bias=False, device='cpu')
ssmblack.load_state_dict(torch.load('iclr_models/ssmblack64.pth', map_location=torch.device('cpu')))

ssmwhite_con = SSM_white_con(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
ssmwhite_con.load_state_dict(torch.load('iclr_models/ssmwhitecon128.pth', map_location=torch.device('cpu')))

ssmgray_con = SSM_gray_con(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
ssmgray_con.load_state_dict(torch.load('iclr_models/ssmgraycon128.pth', map_location=torch.device('cpu')))

ssmblack_con = SSM_black_con(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
ssmblack_con.load_state_dict(torch.load('iclr_models/ssmblackcon128.pth', map_location=torch.device('cpu')))

rnn = RNN(nx, n_m, n_dT, nu, nd, n_hidden, bias=False)
rnn.load_state_dict(torch.load('iclr_models/rnn.pth', map_location=torch.device('cpu')))
building = Building_DAE()

a_matrices = [(np.asarray(building.A), rnn.cells[-1].weight_hh.data.numpy()),
              (ssmblack.A.effective_W().T.detach().cpu().numpy(), ssmgray.A.effective_W().T.detach().cpu().numpy()),
              (ssmwhite.A.effective_W().T.detach().cpu().numpy(), ssmblack_con.A.effective_W().T.detach().cpu().numpy()),
              (ssmgray_con.A.effective_W().T.detach().cpu().numpy(), ssmwhite_con.A.effective_W().T.detach().cpu().numpy())]
plot_matrices(a_matrices, None, 'parameters')

df = pd.DataFrame(index=['True', 'RNN', '$ODE_B$', '$ODE_G$', '$ODE_W$', '$cODE_B$', '$cODE_G$', '$cODE_W$'], columns=['$\lambda_1$', '$\lambda_2$', '$\lambda_3$', '$\lambda_4$'])
a_matrices = [np.asarray(building.A), rnn.cells[-1].weight_hh.data.numpy(),
              ssmblack.A.effective_W().T.detach().cpu().numpy(), ssmgray.A.effective_W().T.detach().cpu().numpy(),
              ssmwhite.A.effective_W().T.detach().cpu().numpy(), ssmblack_con.A.effective_W().T.detach().cpu().numpy(),
              ssmgray_con.A.effective_W().T.detach().cpu().numpy(), ssmwhite_con.A.effective_W().T.detach().cpu().numpy()]
for model, mat in zip(['True', 'RNN', '$ODE_B$', '$ODE_G$', '$ODE_W$', '$cODE_B$', '$cODE_G$', '$cODE_W$'], a_matrices):
    print(model)
    w, v = LA.eig(mat)
    df.loc[model] = w
print(df.to_latex(float_format=lambda x: '%.3f' % x))
