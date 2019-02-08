import projection
import numpy as np
import matplotlib.pyplot as pl
import pyiacsun as ps
import projection
import wavefront as wf

def cmask(a, b, radius, array):    
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius

    return mask

noll0 = 1
radius = 128
alphat = 6*np.pi/4.0
beta = 1.8
t = 0.3
nMax = 4
np.random.seed(123)
a = np.random.randn(nMax)
a[0:-1] = 0.2
a[-1] = 1.0

a = np.array([0.49509614, 0.97544756, 0.18907316, 0.13518821])
metapupil, n, m = wf.zernike(0, npix=int(2*radius))
metapupil *= 0.0
for i in range(nMax):
    z, n, m = wf.zernike(i+noll0, npix=int(2*radius))
    metapupil += a[i] * z

radiusCircle = int(radius / beta)
xCircle = radius + radius * t * np.cos(alphat)
yCircle = radius + radius * t * np.sin(alphat)

circle = pl.Circle((xCircle,yCircle), radiusCircle, fill=False)

xCircleMask = radius + radius * t * np.sin(alphat)
yCircleMask = radius + radius * t * np.cos(alphat)
mask = cmask(xCircleMask, yCircleMask, radiusCircle, metapupil)

maskedMetaPupil = metapupil * mask
maskedMetaPupil = maskedMetaPupil[int(xCircleMask-radiusCircle):int(xCircleMask+radiusCircle),int(yCircleMask-radiusCircle):int(yCircleMask+radiusCircle)]

# M2 = projection.zernikeProjectionMatrix(nMax, beta, t, alphat, includePiston=True)
M2 = projection.zernikeProjectionMatrixNumerical(nMax, beta, t, alphat, includePiston=True, radius=128)
M = np.asarray([[  9.99968057e-01 , -4.24264100e-01 ,  4.24264100e-01 , -8.85698021e-01],
                [  3.18410026e-07 ,  5.55555859e-01 , -3.03239487e-07 , -4.08247739e-01],
                [ -3.18410026e-07 , -3.03239487e-07 ,  5.55555859e-01 ,  4.08247739e-01],
                [  6.64988496e-08 , -3.72670532e-07 ,  3.72670532e-07 ,  3.08642090e-01]])

b = M2 @ a

footprint, n, m = wf.zernike(0, npix=int(2*radiusCircle))
footprint *= 0.0
bCorrect = np.zeros(nMax)
for i in range(nMax):
    z, n, m = wf.zernike(i+noll0, npix=int(2*radiusCircle))
    footprint += b[i] * z
    bCorrect[i] = np.sum(z * maskedMetaPupil) / np.sum(z*z)

print("\n")
print("a   -> ", a)
print("bPR -> ", b)
print("bOK -> ", bCorrect)
print("M -> ")
print(M)
print("M2 -> ")
print(M2)


pl.close('all')
f, ax = pl.subplots(ncols=3, nrows=1, figsize=(16,5))
ps.plot.tvframe(metapupil , ax=ax[0], cmap=pl.cm.jet, vmin=-2, vmax=2)
ax[0].add_artist(circle)
ps.plot.tvframe(maskedMetaPupil, ax=ax[1], cmap=pl.cm.jet, vmin=-2, vmax=2)
ps.plot.tvframe(footprint, ax=ax[2], cmap=pl.cm.jet, vmin=-2, vmax=2)


print(np.max(metapupil * mask)/np.max(footprint))
pl.show()