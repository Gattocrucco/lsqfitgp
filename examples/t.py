import lsqfitgp as lgp
from matplotlib import pyplot as plt
from autograd import numpy as np
import gvar
from scipy import optimize
import time as systime

time = np.arange(21)
x = np.empty((2, len(time)), dtype=[
    ('time', float),
    ('label', int)
])
x['time'][0] = time
delay = 20
x['time'][1] = time - delay
x['label'][0] = 0
x['label'][1] = 1
label_names = ['gatti_comprati', 'gatti_morti']

function = lambda x: np.exp(-1/2 * ((x - 10) / 5)**2)
data_error = 0.05
data_mean = function(x['time']) + data_error * np.random.randn(*x.shape)
data_mean[1] += 0.02 * time
data = gvar.gvar(data_mean, np.full_like(data_mean, data_error))

x = lgp.StructuredArray(x)
def makegp(params):
    kernel = lgp.RatQuad(scale=params['time_scale'], dim='time', alpha=1)
    kernel *= lgp.ExpQuad(scale=params['label_scale'], dim='label')
    gp = lgp.GP(kernel)
    x['time'] = np.array([time, time - params['delay']])
    gp.addx(x, 'A')
    return gp

start = systime.time()
hyperprior = gvar.BufferDict({
    'log(time_scale)': gvar.log(gvar.gvar(10, 10)),
    'log(label_scale)': gvar.log(gvar.gvar(10, 10)),
    'delay': gvar.gvar(10, 20)
})
params = lgp.empbayes_fit(hyperprior, makegp, {'A': data})
end = systime.time()

print('minimization time = {:.2g} sec'.format(end - start))
print('time scale = {}'.format(params['time_scale']))
corr = lgp.ExpQuad(scale=gvar.mean(params['label_scale']))(0, 1)
print('correlation = {:.3g} (equiv. scale = {})'.format(corr, params['label_scale']))
print('delay = {}'.format(params['delay']))

gp = makegp(gvar.mean(params))

xpred = np.empty((2, 100), dtype=x.dtype)
time_pred = np.linspace(np.min(time), np.max(time) + 1.5 * (np.max(time) - np.min(time)), xpred.shape[1])
xpred['time'][0] = time_pred
xpred['time'][1] = time_pred - gvar.mean(params['delay'])
xpred['label'][0] = 0
xpred['label'][1] = 1
gp.addx(xpred, 'B')

pred = gp.predfromdata({'A': data}, 'B')

fig = plt.figure('t')
fig.clf()
ax = fig.subplots(1, 1)

colors = []
for i in range(2):
    m = gvar.mean(pred[i])
    s = gvar.sdev(pred[i])
    polys = ax.fill_between(time_pred, m - s, m + s, alpha=0.5, label=label_names[i])
    colors.append(polys.get_facecolor()[0])

for _, sample in zip(range(3), gvar.raniter(pred)):
    for i in range(2):
        ax.plot(time_pred, sample[i], color=colors[i])

for i in range(2):
    ax.errorbar(time, gvar.mean(data[i]), yerr=gvar.sdev(data[i]), fmt='.', color=colors[i], alpha=1)

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
