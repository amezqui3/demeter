import os
import sys
import glob

from matplotlib import pyplot as plt

import scipy.ndimage as ndimage
import scipy.spatial as spatial
import numpy as np
import pandas as pd

import tifffile as tf

from .misc import clean_zeroes

def normalize_density(img, adjust_by):
    resol = 2**(img.dtype.itemsize*8)
    npz = np.arange(resol, dtype=img.dtype)

    for i in range(len(npz)):
        aux = round(adjust_by[0]*npz[i]*npz[i] + adjust_by[1]*npz[i] + adjust_by[2])
        if aux < resol and aux > 0:
            npz[i] = int(aux)
        elif aux >= resol:
            npz[i] = resol - 1
        else:
            npz[i] = 0

    with np.nditer(img, flags=['external_loop'], op_flags=['readwrite']) as it:
        for x in it:
            x[...] = npz[x]

    return img

def misc_cleaning(img, sigma=3, thr1=55, ero=(7,7,7), dil=(5,5,5), thr2=30, op=(1,11,11)):
    blur = ndimage.gaussian_filter(img, sigma=sigma, mode='constant', truncate=3, cval=0)
    img[blur < thr1] = 0
    print('Gaussian blurred!')
    img = clean_zeroes(img)

    blur = ndimage.grey_erosion(img, mode='constant', size=ero)
    print('Eroded!')

    blur = ndimage.grey_dilation(blur, mode='constant', size=dil)
    print('Dilated!')

    img[blur < thr2] = 0
    blur = ndimage.grey_opening(img, mode='constant', size=op)
    print('Opened!')

    img[blur < thr2-10] = 0
    img = clean_zeroes(img)

    return img

#######################################################################
#######################################################################
#######################################################################

def separate_pruned_spikes(dst, bname, img, cutoff = 1e-2, flex=2):

    labels,num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    print(num,'components')
    regions = ndimage.find_objects(labels)

    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    sz_hist = np.sum(hist)
    print('hist', hist)
    print('size =',sz_hist)

    argsort_hist = np.argsort(hist)[::-1]

    for j in range(len(regions)):
        i = argsort_hist[j]
        r = regions[i]
        if(hist[i]/sz_hist > cutoff):
            z0,y0,x0,z1,y1,x1 = r[0].start,r[1].start,r[2].start,r[0].stop,r[1].stop,r[2].stop
            mask = labels[r]==i+1
            box = img[r].copy()
            box[~mask] = 0
            mass = 1.0/np.sum(box)
            grow = np.arange(box.shape[0], dtype = 'float64')
            grow[0] = np.sum(box[0,:,:])

            for k in range(1,len(grow)):
                zmass = np.sum(box[k,:,:])
                grow[k] = grow[k-1] + zmass

            if grow[-1] != np.sum(box):
                print('grow[-1] != np.sum(box)', j, 'args.in_tiff')
                break

            grow = grow*mass
            logdiff = np.abs(np.ediff1d(np.gradient(np.log(grow))))
            critic = []

            for k in range(len(logdiff)-1,0,-1):
                if(logdiff[k] > 1e-6):
                    critic.append(k)
                    if(len(critic) > flex):
                        break

            if(np.sum(np.ediff1d(critic) == -1) == flex):
                k = critic[0]+1

            print('{} (x,y,z)=({},{},{}), (w,h,d)=({},{},{})'.format(j,x0,y0,z0,box.shape[2],box.shape[1],box.shape[0]))
            print(box.shape[0], critic, logdiff[-1])

            if( k+1 < box.shape[0]):
                print('Reduced from',box.shape[0],'to',k)
                box = box[:k,:,:]

            tf.imwrite('{}{}_l{}_x{}_y{}_z{}.tif'.format(dst,bname,j,x0,y0,z0),box,photometric='minisblack',compress=3)

    return 0

#######################################################################
#######################################################################
#######################################################################

def read_boxes(src):
    barley_files = sorted(glob.glob(src + '*.tif'))
    spikes = len(barley_files)

    if spikes > 5 or spikes < 2:
        print('Found',spikes,'connected components. Expected between 2 and 5. Aborting.')
        sys.exit(0)

    img = dict()
    for i in range(spikes):
        fname = os.path.split(barley_files[i])[1]
        img[fname] = tf.imread(barley_files[i])

    boxes = dict()
    for fname in img:
        coords = []
        fields = os.path.splitext(fname)[0].split('_')
        for f in fields:
            if f[0] in 'lxyz':
                coords.append(f[1:])
        coords = np.array(coords, dtype='int')
        d,h,w = img[fname].shape
        boxes[fname] = (coords[1], coords[2], coords[3], w,h,d, coords[0])

    marker = os.path.split(barley_files[-1])[1]

    return img, boxes, marker

def find_centers(img, boxes, slices=10):
    dcoords = dict()
    acoords = np.empty((len(boxes), 2), dtype=np.float64, order='C')
    for idx,fname in enumerate(boxes):
        xyzcoords = np.nonzero(img[fname][slices,:,:])
        xyzcoords = np.vstack(xyzcoords).T
        foo = np.round(np.mean(xyzcoords, axis=0))
        foo = np.add(foo, np.array([boxes[fname][1], boxes[fname][0]]))
        acoords[idx, :] = foo
        dcoords[fname] = foo

    return acoords, dcoords

def euclidean_dists(centers, marker):
    dummy = dict(zip(centers.keys(), range(len(centers))))
    m_coord = centers[marker]
    euclidean = dict()
    del dummy[marker]

    for fname in dummy:
        euclidean[fname] = np.sqrt(np.sum((m_coord - centers[fname])**2))

    return euclidean


def coloring_spikes(dcenters, euclidean, hull, col_name = ['Red', 'Green', 'Orange', 'Blue']):
    colors = dict()
    red = min(euclidean, key=lambda key: euclidean[key])
    colors[red] = col_name[0]

    foo = hull.points[hull.vertices]

    idx_r = 0
    for i in range(foo.shape[0]):
        if np.sum(dcenters[red] == foo[i]) == foo.shape[1]:
            idx_r = i

    for i in range(1,foo.shape[0]):
        for fname,xy in dcenters.items():
            if np.sum(xy == foo[(idx_r+i)%foo.shape[0]]) == foo.shape[1]:
                colors[fname] = col_name[i]

    return colors

def render_alignment(dst, sname, boxes, colors, write_fig=False):
    fig = plt.figure(figsize=(10,6))
    fig.suptitle(sname, fontsize=25)
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=80., azim=40)

    for fname in boxes:
        x,y,z,w,h,d,l = boxes[fname]
        cx,cy = x+w/2,y+h/2
        ax.plot([x,x,x,x,x+w,x+w,x+w,x+w,x],[y,y,y+h,y+h,y+h,y+h,y,y,y],[z,z+d,z+d,z,z,z+d,z+d,z,z],c=colors[fname].lower())
        ax.text(x+w/2,y+h/2,z+d,'{} {}'.format('L', l),None, color=colors[fname].lower(), ha='center', va='center')

    if write_fig:
        plt.savefig(dst+sname+'_alignment.jpg', dpi=150, format='jpg', pil_kwargs={'optimize':True})

def barley_labeling(src, dst, rename=True):
    img,boxes,marker = read_boxes(src)

    centers,dcenters = find_centers(img, boxes)
    hull = spatial.ConvexHull(centers[:-1,:])

    if len(hull.vertices) != 4:
        print(src,'\nConvex Hull reports less than 4 spikes!!!.')
        sys.exit(0)

    euclidean = euclidean_dists(dcenters, marker)
    colors = coloring_spikes(dcenters, euclidean, hull)
    colors[marker] = 'Black'

    sname = marker.split('_')[0]
    render_alignment(dst, sname, boxes,colors)
    braces = '{}_'

    if rename:
        for fname in colors:
            splt = fname.split('_')
            cname = braces*len(splt) + '{}'
            cname = cname.format(sname,splt[1],colors[fname],*(splt[2:]))
            os.rename(src + fname, src + cname)

    return dst, sname, boxes, colors


#######################################################################
#######################################################################
#######################################################################

def seed_template(dst, img, seed, loc, size, padding=7, dil=4, thr=120, op=5, ero=3, sigma= 3, bname='seed', w=False):
    x,y,z = loc
    w,h,d = size

    pad_array = np.array([padding,padding,padding])

    for i in range(3):
        if loc[i] < padding:
            pad_array[i] = loc[i]

    pad_array[0], pad_array[-1] = pad_array[-1], pad_array[0]

    if z+d+pad_array[0] > img.shape[0]:
        pad_array[0] -= (z+d+pad_array[0]) -  img.shape[0]

    if y+h+pad_array[1] > img.shape[1]:
        pad_array[1] -= (y+h+pad_array[1]) -  img.shape[1]

    if x+w+pad_array[2] > img.shape[2]:
        pad_array[2] -= (x+w+pad_array[2]) -  img.shape[2]

    padded = np.zeros((np.array(seed.shape) + 2*pad_array)).astype('uint8')
    foo = np.array(seed.shape) + pad_array
    padded[ pad_array[0]:foo[0], pad_array[1]:foo[1], pad_array[2]:foo[2] ] = seed

    for i in range(dil):
        padded = ndimage.grey_dilation(padded,
                                       structure=(ndimage.generate_binary_structure(padded.ndim, 1)),
                                       mode='constant', cval=0)

    iso_seed = img[(z-pad_array[0]):(z+d+pad_array[0]),
                   (y-pad_array[1]):(y+h+pad_array[1]),
                   (x-pad_array[2]):(x+w+pad_array[2])].copy()

    padmask = padded > 50
    iso_seed = np.where(padmask, iso_seed, 0)
    iso_seed[iso_seed < thr] = 0
    iso_seed = ndimage.grey_opening(iso_seed, size=(dil,dil,dil), mode='constant', cval = 0)

    if ero > 1:
        iso_seed = ndimage.grey_erosion(iso_seed, size=(ero,ero,ero), mode='constant', cval = 0)

    labels,num = ndimage.label(iso_seed, structure=ndimage.generate_binary_structure(img.ndim, 1))
   # print(num,'components')
    regions = ndimage.find_objects(labels)

    if num > 1:
        hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
        argsort_hist = np.argsort(hist)[::-1]
        i = argsort_hist[0]
        r = regions[i]
        mask = labels[r] == i+1
        box = iso_seed[r].copy()
        box[~mask] = 0
        iso_seed = box.copy()

    gauss = ndimage.gaussian_filter(iso_seed, sigma=sigma, mode='constant', cval=0, truncate=4)
    iso_seed[gauss < 0.75*thr] = 0

    if w:
        outname = '{}{}_p{}_d{}_t{}_o{}_e{}_g{}.tif'.format(dst,bname,padding, dil, thr, op, ero, sigma)
        tf.imwrite(outname, iso_seed, photometric='minisblack', compress=3)


    return iso_seed

def refine_pesky_seeds(dst, img, cutoff=1e-2, opening=7, write_tif=False, median=1e5, med_range=2000, bname='test'):
    split_further = False
    tol = 1.5
    img = ndimage.grey_opening(img, size=(opening, opening, opening), mode='constant', cval = 0)
    sizes = []
    locs = []

    counter = 0
    labels,num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    print(num,'components')

    regions = ndimage.find_objects(labels)
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    sz_hist = ndimage.sum(hist)
    argsort_hist = np.argsort(hist)[::-1]
    print('hist', hist[argsort_hist])
    print('size =',sz_hist)

    if num > 1:

        for j in range(len(regions)):
            i = argsort_hist[j]
            r = regions[i]
            if (hist[i]/sz_hist > cutoff) and (np.fabs(hist[i] - median) < tol*med_range):
                z0,y0,x0,z1,y1,x1 = r[0].start,r[1].start,r[2].start,r[0].stop,r[1].stop,r[2].stop
                mask = labels[r]==i+1
                box = img[r].copy()
                box[~mask] = 0
                print('{}\t(x,y,z)=({},{},{}),\t (w,h,d)=({},{},{})'.format(j,x0,y0,z0,box.shape[2],box.shape[1],box.shape[0]))
                sizes.append((box.shape[2],box.shape[1],box.shape[0]))
                locs.append((x0,y0,z0))
                if write_tif:
                    tf.imwrite('{}{}_{}.tif'.format(dst,bname,counter),box,photometric='minisblack',compress=3)
                counter += 1

                print('---\n')

            elif np.fabs(hist[i] - median) >= tol*med_range:
                print('seed', bname,'_',j,' is too large/small.')
                split_further = True


    else:
        print(bname, 'could not be broken up further.')
        split_further = True

    return locs, sizes, split_further

def open_decompose(dst, box, cutoff=1e-2, opening = 7, write_tif=False, bname='test'):
    sizes = []
    locs = []
    box = ndimage.grey_opening(box, size=(opening, opening, opening), mode='constant', cval = 0)

    olabels,onum = ndimage.label(box, structure=ndimage.generate_binary_structure(box.ndim, 1))

    oregions = ndimage.find_objects(olabels)
    ohist,obins = np.histogram(olabels, bins=onum, range=(1,onum+1))
    osz_hist = ndimage.sum(ohist)
    oargsort_hist = np.argsort(ohist)[::-1]
    for k in range(len(oregions)):
        l = oargsort_hist[k]
        r = oregions[l]
        if(ohist[l]/osz_hist > cutoff):
            z0,y0,x0,z1,y1,x1 = r[0].start,r[1].start,r[2].start,r[0].stop,r[1].stop,r[2].stop
            omask = olabels[r]==l+1
            obox = box[r].copy()
            obox[~omask] = 0
            sizes.append((obox.shape[2],obox.shape[1],obox.shape[0]))
            locs.append((x0,y0,z0))
            if write_tif:
                tf.imwrite('{}{}_comp_{}.tif'.format(dst,bname,k),obox,photometric='minisblack',compress=3)

    return locs, sizes, ohist[oargsort_hist]

def preliminary_extract(src, seeddst, figdst, fname, bname, cutoff=1e-2, threshold = 200, write_tif=False, med_tol=0.5, op=7):

    img = tf.imread(src+fname)

    sname = os.path.splitext(fname)[0]
    sname = '_'.join(sname.split('_')[0:3])

    img[img < threshold] = 0
    locs, sizes, hist = open_decompose(seeddst, img, bname=bname, write_tif=write_tif, opening=op)

    diff = np.abs(np.ediff1d(hist))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    med = np.median(hist)
    med_cut = med_tol*med

    axes[0].axhline(med, c='m', lw=3, label='median')
    axes[0].axhline(med+med_cut, c='m', lw=3, ls=':', label='median + {:.0f}%'.format(100*med_tol))
    axes[0].axhline(med-med_cut, c='m', lw=3, ls=':')

    axes[0].plot(hist, color='blue', marker='o', lw=0, ms=9)
    axes[0].set_xlabel('seed', fontsize=18)
    axes[0].set_ylabel('volume', fontsize=18)
    axes[0].legend(fontsize=15)

    axes[1].axhline(med_cut, c='m', lw=3, ls=':', label='{:.0f}% median'.format(100*med_tol))
    axes[1].plot(diff, color='red', marker='o', lw=0, ms=9)
    axes[1].set_xlabel('seed', fontsize=18)
    axes[1].set_ylabel('| change in size |', fontsize=18)
    axes[1].legend(fontsize=15)
    fig.suptitle(sname, fontsize=24);

    fig.savefig(figdst + sname + '_preliminary.jpg', dpi=100, format='jpg', pil_kwargs={'optimize':True})

    to_split = []
    to_ignore = []

    for i in range(len(diff)):
        if diff[i] > med_cut  and hist[i] - med > med_cut :
            to_split.append(i+1)
        if diff[i] > med_cut  and med - hist[i+1] > med_cut :
            to_ignore.append(i+1)

    print(len(to_split), to_split)
    print(len(to_ignore), to_ignore)

    if len(to_split) == 0:
        start_seed = 0
    else:
        start_seed = to_split[-1]

    if len(to_ignore) == 0:
        end_seed = len(locs)
    else:
        end_seed = to_ignore[0]

    return locs, sizes, med, med_cut, start_seed, end_seed

def extract_refinement(dst, bname, med, med_cut, start_seed, end_seed, threshold=200, op=7, iter_tol=3, w=False):
    locs0 = []
    sizes0 = []
    to_ignore = []

    if start_seed > 0:
        for i in range(start_seed):
            iter0 = 0
            threshold0 = threshold + 5
            opening0 = op
            split_further = True

            l_name = bname + '_comp_{}.tif'.format(i)
            l_seed = tf.imread(dst + l_name)

            while ((split_further == True) and (iter0 < iter_tol)):
                print(i, '\tthreshold:',threshold0, '\topening:', opening0)
                l_seed[ l_seed < threshold0 ] = 0
                locs_temp, sizes_temp,split_further = refine_pesky_seeds(dst, l_seed, cutoff=1e-2, opening=opening0,
                                                                         write_tif=w, median=med,
                                                                         med_range=med_cut,
                                                                         bname=os.path.splitext(l_name)[0])
                iter0 += 1
                if iter0 % 2 == 0:
                    threshold0 +=2
                else:
                    opening0 +=1

                print('#################')

            if split_further:
                to_ignore.append(i)

            locs0.append(locs_temp)
            sizes0.append(sizes_temp)

    return locs0, sizes0, to_ignore

def seed_reconstruction(src, dst, fname, locs, sizes, locs0, sizes0, start_seed, end_seed, to_ignore,
                        padding=7, dil=4, thr=120, op=7, ero=1, sigma= 3, write_file=False):

    img = tf.imread(src+fname)
    bname = os.path.splitext(fname)[0]
    bname = '_'.join(bname.split('_')[1:3])

    if start_seed > 0:
        for i in range(start_seed):
            if i not in to_ignore:
                seed_files = sorted(glob.glob(dst + bname + '_comp_{}_*'.format(i)))
                for j in range(len(seed_files)):
                    seed = tf.imread(seed_files[j])
                    seed_loc = (locs[i][0] + locs0[i][j][0], locs[i][1] + locs0[i][j][1], locs[i][2] + locs0[i][j][2])
                    seed_size = sizes0[i][j]
                    seed_name = 'seed_{}_{}'.format(i,j)
                    iso_seed = seed_template(dst, img, seed, seed_loc,seed_size, padding=padding, op=op, dil=dil,
                                             ero=ero, thr=thr, sigma=sigma, bname=seed_name, w=write_file)

    for i in range(start_seed, end_seed, 1):
        seed = tf.imread(dst + bname + '_comp_{}_0.tif'.format(i))
        pos , size = locs[i], sizes[i]
        seed_name = 'seed_{}_0'.format(i)
        iso_seed = seed_template(dst, img, seed, pos, size, padding=padding, op=op, dil=dil,
                                 ero=ero, thr=thr, sigma=sigma, bname=seed_name, w=write_file)

    if end_seed < len(locs):
        for i in range(end_seed, len(locs)):
            seed = tf.imread(dst + bname + '_comp_{}_0.tif'.format(i))
            pos , size = locs[i], sizes[i]
            seed_name = 'seed_{}_0'.format(i)
            iso_seed = seed_template(dst, img, seed, pos, size, padding=padding, op=op, dil=dil+2,
                                     ero=ero, thr=thr, sigma=sigma, bname=seed_name, w=write_file)

def seed_isolation(src, figdst, fname, cutoff=1e-2, threshold = 200, med_tol=0.5,
                   padding=7, dil=4, thr=120, op=7, ero=1, sigma= 3, iter_tol=3, write_file=True):

    bname = os.path.splitext(fname)[0]
    bname = '_'.join(bname.split('_')[1:3])
    dst = src + bname + '_seeds/'

    if os.path.isdir(dst):
        pass
    else:
        os.makedirs(dst)
        print('directory', bname + '_seeds created')

    locs,sizes,med,med_cut,start_seed,end_seed = preliminary_extract(src,dst,figdst,fname,bname, threshold = threshold,
                                                                     write_tif=write_file,med_tol=med_tol, op=op)

    print('\n#####################\n')
    print('median size = ', med, '. 50% = ', med_cut)
    print('[',med+med_cut,'<->',med-med_cut,']')
    print('start = ', start_seed,'; end = ', end_seed)
    print('\n^^^^^^^^^^^^^^\n')

    locs0, sizes0, to_ignore = extract_refinement(dst, bname, med, med_cut, start_seed, end_seed,
                                       threshold=threshold, op=op, iter_tol=iter_tol, w=write_file)

    for i in range(start_seed, len(locs), 1):
        seed_file0 = dst + bname + '_comp_{}.tif'.format(i)
        seed_file  = dst + bname + '_comp_{}_0.tif'.format(i)
        os.rename(seed_file0, seed_file)

    seed_reconstruction(src, dst, fname, locs, sizes, locs0, sizes0, start_seed, end_seed, to_ignore,
                        padding=padding, dil=dil, thr=thr, op=op, ero=ero, sigma=sigma,
                        write_file=write_file)

    return locs, sizes, start_seed, end_seed, to_ignore


def missing_summary(scanId, color, locs, sizes, to_ignore):
    missing_seed = np.zeros((len(to_ignore), 4), np.float32)

    for i in range(len(to_ignore)):
        missing_seed[i, :] = [to_ignore[i],
                              locs[to_ignore[i]][0] + 0.5*sizes[to_ignore[i]][0],
                              locs[to_ignore[i]][1] + 0.5*sizes[to_ignore[i]][1],
                              locs[to_ignore[i]][2] + 0.5*sizes[to_ignore[i]][2]]

    bar = (pd.DataFrame([scanId, color])).T
    bar.columns = ['scan', 'color']
    bar = pd.concat([bar]*len(to_ignore))
    bar.index = list(range(len(to_ignore)))

    df = pd.DataFrame(missing_seed, columns=['seedNo', 'centerX', 'centerY', 'centerZ'])

    foo = pd.merge(bar,df, left_index=True, right_index=True).astype({'seedNo': 'uint8'})

    return foo

#######################################################################
#######################################################################
#######################################################################
