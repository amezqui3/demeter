import itertools

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm

import scipy.ndimage as ndimage
import scipy.special as special

import numpy as np
import pandas as pd

import tifffile as tf

def neighborhood_setup(dimension):
    neighs = sorted(list(itertools.product(range(2), repeat=dimension)), key=np.sum)[1:]
    subtuples = dict()
    for i in range(len(neighs)):
        subtup = [0]
        for j in range(len(neighs)):
            if np.all(np.subtract(neighs[i], neighs[j]) > -1):
                subtup.append(j+1)
        subtuples[neighs[i]] = subtup

    return neighs, subtuples

def neighborhood(voxel, neighs, hood, dcoords):
    hood[0] = dcoords[voxel]
    neighbors = np.add(voxel, neighs)
    for j in range(1,len(hood)):
        key = tuple(neighbors[j-1,:])
        if key in dcoords:
            hood[j] = dcoords[key]
    return hood

def centerVertices(verts):
    origin = -1*np.mean(verts, axis=0)
    verts = np.add(verts, origin)
    return verts

class CubicalComplex:

    def __init__(self, img):
        self.img = img

    def complexify(self, center=True):
        coords = np.nonzero(self.img)
        coords = np.vstack(coords).T
        keys = [tuple(coords[i,:]) for i in range(len(coords))]
        dcoords = dict(zip(keys, range(len(coords))))
        neighs, subtuples = neighborhood_setup(self.img.ndim)
        binom = [special.comb(self.img.ndim, k, exact=True) for k in range(self.img.ndim+1)]

        hood = np.zeros(len(neighs)+1, dtype=np.int)-1
        cells = [[] for k in range(self.img.ndim+1)]

        for voxel in dcoords:
            hood.fill(-1)
            hood = neighborhood(voxel, neighs, hood, dcoords)
            nhood = hood > -1
            c = 0
            if np.all(nhood[:-1]):
                for k in range(1, self.img.ndim):
                    for j in range(binom[k]):
                        cell = hood[subtuples[neighs[c]]]
                        cells[k].append(cell)
                        c += 1
                if nhood[-1]:
                    cells[self.img.ndim].append(hood.copy())
            else:
                for k in range(1, self.img.ndim):
                    for j in range(binom[k]):
                        cell = nhood[subtuples[neighs[c]]]
                        if np.all(cell):
                            cells[k].append(hood[subtuples[neighs[c]]])
                        c += 1

        dim = self.img.ndim
        for k in range(dim, -1, -1):
            if len(cells[k]) > 0:
                break

        self.ndim = dim
        self.cells = [np.array(cells[k]) for k in range(dim+1)]
        if center:
            self.cells[0] = centerVertices(coords)
        else:
            self.cells[0] = coords

        return self

    def EC(self):
        chi = 0
        for i in range(len(self.cells)):
            chi += ((-1)**i)*len(self.cells[i])

        self.chi = chi
        return chi

    def summary(self):
        cellnames = ['vertices', 'edges', 'squares', 'cubes']
        for i in range(len(self.cells)):
            if i < len(cellnames):
                print('{}\t{}'.format(len(self.cells[i]), cellnames[i]))
            else:
                print('{}\t{:02d}hypercubes'.format(len(self.cells[i]), i))

        chi = self.EC()
        print('----\nEuler Characteristic: {}'.format(chi))
        return 0

    def ECC(self, filtration, T=32, bbox=None):

        if bbox is None:
            minh = np.min(filtration)
            maxh = np.max(filtration)
        else:
            minh,maxh = bbox

        buckets = [None for i in range(len(self.cells))]

        buckets[0], bins = np.histogram(filtration, bins=T, range=(minh, maxh))

        for i in range(1,len(buckets)):
            if len(self.cells[i]) > 0 :
                buckets[i], bins = np.histogram(np.max(filtration[self.cells[i]], axis=1), bins=T, range=(minh, maxh))

        ecc = np.zeros_like(buckets[0])
        for i in range(len(buckets)):
            if buckets[i] is not None:
                ecc = np.add(ecc, ((-1)**i)*buckets[i])

        return np.cumsum(ecc)

    def ECT(self, directions, T=32, verts=None, bbox=None):
        if verts is None:
            verts = self.cells[0]

        ect = np.zeros(T*directions.shape[0], dtype=int)

        for i in range(directions.shape[0]):
            heights = np.sum(verts*directions[i,:], axis=1)
            ecc = self.ECC(heights, T, bbox)
            ect[i*T : (i+1)*T] = ecc

        return ect

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

def pole_directions(parallels, meridians, x=0, y=1, z=2, tol=1e-10):
    dirs = np.zeros((2*(meridians*parallels)-meridians+2, 3), dtype=np.float64)
    idx = 1

    dirs[0, :] = np.array([0,0,0])
    dirs[0, z] = 1

    for i in range(parallels):
        theta = (i+1)*np.pi/(2*parallels)
        for j in range(meridians):
            phi = j*2*np.pi/meridians
            dirs[idx,x] = np.cos(phi)*np.sin(theta)
            dirs[idx,y] = np.sin(phi)*np.sin(theta)
            dirs[idx,z] = np.cos(theta)
            idx += 1

    for i in range(parallels-1):
        theta = (i+1)*np.pi/(2*parallels) + 0.5*np.pi
        for j in range(meridians):
            phi = j*2*np.pi/meridians
            dirs[idx,x] = np.cos(phi)*np.sin(theta)
            dirs[idx,y] = np.sin(phi)*np.sin(theta)
            dirs[idx,z] = np.cos(theta)
            idx += 1


    dirs[-1, :] = np.array([0,0,0])
    dirs[-1, z] = -1
    dirs[np.abs(dirs) < tol] = 0

    return dirs

def random_directions(dims=3, N=50, r=1):
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2*np.pi, N)

    if dims == 2:
        x = r*np.cos(phi)
        y = r*np.sin(phi)

        return np.column_stack((x,y))

    if dims == 3:
        z = rng.uniform(-r, r, N)
        x = np.sqrt(r**2 - z**2)*np.cos(phi)
        y = np.sqrt(r**2 - z**2)*np.sin(phi)

        return np.column_stack((x,y,z))

    else:
        print("Function implemented only for 2 and 3 dimensions")

def regular_directions(dims=3, N=50, r=1):
    if dims==2:
        eq_angles = np.linspace(0, 2*np.pi, num=N, endpoint=False)
        return np.column_stack((np.cos(eq_angles), np.sin(eq_angles)))

    if dims==3:
        dirs = np.zeros((N, 3), dtype=np.float64)
        i = 0
        a = 4*np.pi*r**2/N
        d = np.sqrt(a)
        Mtheta = np.round(np.pi/d)
        dtheta = np.pi/Mtheta
        dphi = a/dtheta
        for m in range(int(Mtheta)):
            theta = np.pi*(m + 0.5)/Mtheta
            Mphi = np.round(2*np.pi*np.sin(theta)/dphi)
            for n in range(int(Mphi)):
                phi = 2*np.pi*n/Mphi

                dirs[i,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
                i += 1

        return dirs
    else:
        print("Function implemented only for 2 and 3 dimensions")

def plot_pole_directions(directions, titleplot = 'title', parallels=8, meridians=12, save_fig=False, dst = './', filename = 'sphere'):

    pdirections = pole_directions(parallels,meridians,x=1,y=0,z=2)
    viridis = cm.get_cmap('viridis', parallels*2-1)
    opacity = np.linspace(0.25,0.9,parallels*2-1)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    axlen = np.linspace(np.min(directions), np.max(directions), 50)
    zeros = np.repeat(0,len(axlen))

    ax.plot(axlen, zeros, zeros, c='r', lw=4, alpha=0.5)
    ax.plot(zeros, axlen, zeros, c='b', lw=4, alpha=0.5)
    ax.plot(zeros, zeros, axlen, c='g', lw=4, alpha=0.5)
    ax.scatter(directions[:,0],directions[:,1],directions[:,2], marker='^', s=160, c='m')

    for i in range(parallels*2-1):
        ax.plot(pdirections[(i*meridians + 1):((i+1)*meridians + 1),0],
                pdirections[(i*meridians + 1):((i+1)*meridians + 1),1],
                pdirections[(i*meridians + 1):((i+1)*meridians + 1),2],
                c=viridis.colors[i,:3],
                alpha = opacity[i],
                lw = 2.5)
        ax.plot(pdirections[np.array((i*meridians + 1,(i+1)*meridians)),0],
                pdirections[np.array((i*meridians + 1,(i+1)*meridians)),1],
                pdirections[np.array((i*meridians + 1,(i+1)*meridians)),2],
                c=viridis.colors[i,:3],
                alpha = opacity[i],
                lw = 2.5)

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_axis_off()
    #ax.set_xlabel('X Axis', fontsize=24)
    #ax.set_ylabel('Y Axis', fontsize=24)
    #ax.set_zlabel('Z Axis', fontsize=24)
    ax.set_title(titleplot,fontsize=40, pad=0, y=0.83);
    #plt.margins(0.99)
    #ax = plt.gca()
    #ax.set_xlim(0.0, 1.0);
    #ax.set_ylim(1.0, 0.0);

    if save_fig:
        plt.savefig(dst+filename+'.jpg', bbox_inches='tight',
                    dpi=72, format='jpg', pil_kwargs={'optimize':True});
        plt.savefig(dst+filename+'.pdf', dpi=72,
                    bbox_inches='tight', format='pdf');
        plt.close()
