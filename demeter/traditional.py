import scipy.ndimage as ndimage
from scipy.spatial import ConvexHull
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

import pandas as pd
import tifffile as tf

from .misc import rotateSVD, find_tip


def load_metadata(meta,scan_name,bcolor,seed_no,spike_label,columns=[]):

    scan_meta = meta[meta['Scan'] == scan_name]
    spike_meta = scan_meta[scan_meta['Color'] == bcolor]

    spike_meta = pd.concat([spike_meta]*seed_no)

    spike_meta['Label'] = spike_label

    if len(columns) > 0:
        for col in columns:
            spike_meta[col] = 1.0

    return spike_meta

def seed_lengths(seed, x=0,y=1,z=2):
    maxs, mins = np.max(seed, axis=0), np.min(seed, axis=0)
    length, width, heightmax = maxs - mins
    height = np.max(seed[ np.abs(seed[:,y]) < 0.5 ], axis=0)[z] - mins[z]

    return length, width, height, heightmax


def seed_area_vol(img, coords, border):
    surface = ndimage.convolve(img, border, np.int8, 'constant', cval=0)

    surface[ surface < 0 ] = 0
    area_sq = np.sum(surface)
    area_cube = np.sum(surface > 0)
    vol = np.sum(img)

    hull = ConvexHull(coords)

    area_ratio = area_sq/hull.area
    vol_ratio = vol/hull.volume

    return area_sq,area_cube,vol,hull.area,hull.volume,area_ratio,vol_ratio

def save_alignment(dst, bname, seed, sigma):
    np.savetxt(dst+bname+'_sigma.csv', sigma, fmt='%.5e', delimiter = ',')
    np.savetxt(dst+bname+'_coords.csv', seed, fmt='%.5e', delimiter = ',')

def traditional_summary(dst, csv_path, scan_path, color_list, border_mask, save_coords=False):

    scan_name = os.path.normpath(scan_path).split(os.path.sep)[-1]
    x,y,z = 2,1,0
    traits = ['Length', 'Width', 'Height', 'HeightMax', 'Shell', 'Area', 'Vol', 'ConvexArea', 'ConvexVol', 'ConvexAreaRatio', 'ConvexVolRatio']

    meta = pd.read_csv(csv_path)

    for color in color_list:
        print('***********************************')
        print(scan_name, color)
        seed_files = glob.glob(scan_path + '*' + color + '*/seed_*_p*.tif')
        Tag, Length,Width,Height,HeightMax,Shell,Area,Vol,ConvexArea,ConvexVol,ConvexAreaRatio,ConvexVolRatio = [],[],[],[],[],[],[],[],[],[],[],[]

        if len(seed_files) < 1:
            print('Seeds not found in ', scan_path)

        else:

            csvdst = dst + scan_name + '/'
            if not os.path.isdir(csvdst):
                os.makedirs(csvdst)

            foodst = csvdst + color + '/'
            if save_coords and not os.path.isdir(foodst):
                os.makedirs(foodst)

            src, fname = os.path.split(seed_files[0])
            label = ((os.path.normpath(src).split(os.path.sep)[-1]).split('_')[0])[-1]

            summary = load_metadata(meta,scan_name,color, len(seed_files), label, traits)

            for seed_file in seed_files:

                raw, fname = os.path.split(seed_file)
                bname = '_'.join(os.path.splitext(fname)[0].split('_')[:3])

                img = tf.imread(seed_file)
                img[img > 0] = 1
                img = ndimage.binary_fill_holes(img).astype(img.dtype)
                img = ndimage.binary_closing(img, iterations=2, border_value=0).astype(img.dtype)
                img = ndimage.binary_fill_holes(img).astype(img.dtype)

                coords = np.vstack(np.nonzero(img)).T
                origin = -1*np.mean(coords, axis=0)
                coords = np.add(coords, origin)

                max_vox = find_tip(coords, x,y,z)
                seed, sigma, vh, rotZ, rotX = rotateSVD(coords, max_vox)

                length, width, height, heightmax = seed_lengths(seed)
                area_sq,area_cube,vol,hullarea,hullvol,area_ratio,vol_ratio = seed_area_vol(img, coords, border_mask)

                Tag.append(bname)
                Length.append(length)
                Width.append(width)
                Height.append(height)
                HeightMax.append(heightmax)
                Area.append(area_sq)
                Shell.append(area_cube)
                Vol.append(vol)
                ConvexArea.append(hullarea)
                ConvexVol.append(hullvol)
                ConvexAreaRatio.append(area_ratio)
                ConvexVolRatio.append(vol_ratio)

                if save_coords:
                    save_alignment(foodst, bname, seed, sigma)


            summary['Tag'] = Tag
            summary['Length'] = Length
            summary['Width'] = Width
            summary['Height'] = Height
            summary['HeightMax'] = HeightMax
            summary['Shell'] = Shell
            summary['Area'] = Area
            summary['Vol'] = Vol
            summary['ConvexArea'] = ConvexArea
            summary['ConvexVol'] = ConvexVol
            summary['ConvexAreaRatio'] = ConvexAreaRatio
            summary['ConvexVolRatio'] = ConvexVolRatio

            summary.to_csv(csvdst + scan_name + '_' + color +'_summary.csv', index=False)

    return summary
