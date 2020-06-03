import matplotlib.pyplot as plt
import numpy as np
import pandas


steps = [8, 16, 32, 64, 128, 256]
models = ['RNN', 'SSMBLACK', 'SSMGRAY', 'SSMWHITE', 'SSMBLACKCON', 'SSMGRAYCON', 'SSMWHITECON']
stdopen, stdnstep, meanopen, meannstep, minopen, minnstep = [pandas.DataFrame(index=models,
                                                                              columns=steps)
                                                             for i in range(6)]
for n in models:
    for step in steps:
        print(f'{n.lower()}_{step}.csv')
        res = pandas.read_csv(f'iclr_results/csvs/{n.lower()}_{step}.csv')
        if 'dev_openloss' in res.columns:
            best = res.loc[res['dev_openloss'].idxmin()]
            minnstep.loc[n][int(step)] = best['nstep_test_loss']
            minopen.loc[n][int(step)] = best['Test_openloss']
            res = res.loc[res['lr'] == best['lr']]
            res = res.loc[res['Test_openloss'].notnull()]
            nsteploss = res['nstep_test_loss']
            openloss = res['Test_openloss']
            mean_openloss = openloss.mean()
            mean_nsteploss = nsteploss.mean()
            std_openloss = openloss.std()
            std_nsteploss = nsteploss.std()
            stdopen.loc[n][int(step)] = std_openloss
            stdnstep.loc[n][int(step)] = std_nsteploss
            meanopen.loc[n][int(step)] = mean_openloss
            meannstep.loc[n][int(step)] = mean_nsteploss


for k in [stdopen, stdnstep, meanopen, meannstep, minopen, minnstep]:
    print(k.to_latex(float_format=lambda x: '%.3f' % x))

ind = np.arange(6)
labels = ['ODE$_B$', 'ODE$_G$', 'ODE$_W$', 'cODE$_B$', 'cODE$_G$', 'cODE$_W$']
markers = ['x', '*', '+', 'o', 'v', 's']
lines = ['--', '-.', '-', ':', '--', '-.']
fig, ax = plt.subplots() # create a new figure with a default 111 subplot
print(np.array(minopen.loc['SSMBLACK']))
for n, label, m, l in zip(models[1:], labels, markers, lines):
    ax.plot(np.arange(5), np.array(minopen.loc[n])[:-1], label=label, marker=m, linestyle=l)

plt.xlabel('Training Prediction Horizon')
plt.ylabel('Open loop MSE')
plt.xticks(range(5), steps[:-1])
plt.legend()
plt.tight_layout()
plt.savefig('open_mse_line.pdf')
plt.savefig('open_mse_line.png')

fig, ax = plt.subplots()
for n, label, m, l in zip(models[1:], labels, markers, lines):
    ax.plot(np.arange(5), np.array(minnstep.loc[n])[:-1], label=label, marker=m, linestyle=l)

plt.xlabel('Training Prediction Horizon')
plt.ylabel('N-step MSE')
plt.xticks(range(5), steps[:-1])
plt.legend()
plt.tight_layout()
plt.savefig('nstep_mse_line.pdf')
plt.savefig('nstep_mse_line.png')
