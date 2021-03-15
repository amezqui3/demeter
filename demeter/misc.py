import itertools

import numpy as np

from matplotlib import pyplot as plt


def clean_zeroes(img):
    d = img.ndim
    zeroes = []
    for ax in itertools.combinations(np.arange(d), d-1):
        row = np.any(img != 0, axis=ax)
        bar = np.nonzero(np.ediff1d(1-row))[0]
        row[bar] = True
        row[(bar+1)] = True
        zeroes.append(row)

    if d == 1:
        return img[zeroes[0]]

    mask = np.tensordot(zeroes[-1], zeroes[-2], axes=0)
    for i in range(3,d+1):
        mask = np.tensordot(mask, zeroes[-i], axes=0)

    shape = tuple([np.sum(zeroes[-i]) for i in range(1,len(zeroes)+1)])

    return img[mask].reshape(shape)

def find_tip(coords, x,y,z):
    maxes = np.max(coords, axis=0)
    max_vox = coords[coords[:, z] == maxes[z]]
    if len(max_vox) > 1 :
        maxesz = np.max(max_vox, axis=0)
        max_vox = max_vox[max_vox[:, y] == maxesz[y]]

        if len(max_vox) > 1:
            maxesy = np.max(max_vox, axis=0)
            max_vox = max_vox[max_vox[:, x] == maxesy[x]]

    return np.squeeze(max_vox)

def rotateSVD(coords, max_vox, x=0,y=1,z=2):
    u, s, vh = np.linalg.svd(coords, full_matrices=False)
    sigma = np.sqrt(s)
    seed = np.matmul(coords, np.transpose(vh))
    y_pos = seed[seed[:,y] > 0]
    y_neg = seed[seed[:,y] < 0]

    y_posmax = np.max(y_pos, axis=0)
    y_posmin = np.min(y_pos, axis=0)
    y_negmax = np.max(y_neg, axis=0)
    y_negmin = np.min(y_neg, axis=0)
    hzp = np.squeeze(y_pos[y_pos[:,z]==y_posmax[z]])[y] - np.squeeze(y_neg[y_neg[:,z]==y_negmax[z]])[y]
    hzn = np.squeeze(y_pos[y_pos[:,z]==y_posmin[z]])[y] - np.squeeze(y_neg[y_neg[:,z]==y_negmin[z]])[y]

    rotZ = False
    if hzn > hzp:
        seed[:, z] = -1.0*seed[:, z]
        rotZ = True

    rotX = False
    if max_vox[0] < 0:
        seed[:, x] = -1.0*seed[:, x]
        rotX = True

    return seed, sigma, vh, rotZ, rotX

def plot_3Dprojections(seed, title='title', markersize=2, writefig=False, dst='./', dpi=150):
    axes = ['X','Y','Z']
    fig, ax = plt.subplots(1,3,figsize=(12,4))

    for i in range(3):
        proj = []
        for j in range(3):
            if j != i:
                proj.append(j)
        ax[i].plot(seed[:,proj[0]], seed[:,proj[1]], '.', ms=markersize, c='y')
        ax[i].set_xlabel(axes[proj[0]])
        ax[i].set_ylabel(axes[proj[1]])
        ax[i].set_title(axes[i] + ' Projection')
        ax[i].set_aspect('equal');

    fig.suptitle(title, y=0.95, fontsize=20)
    plt.tight_layout();

    if writefig:
        filename = '_'.join(title.split(' ')).lower()
        plt.savefig(dst + filename + '.png', dpi=dpi, format='png', bbox_inches='tight',
                    facecolor='white', transparent=False)
        plt.close();
