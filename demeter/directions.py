import numpy as np

import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt

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
