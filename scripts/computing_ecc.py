import tifffile as tf
import numpy as np
import os
import argparse
import glob
import importlib
import scipy.ndimage as ndimage
import pandas as pd
import brewing_utils as brew

def EulerBrew(dst, meta_path, scan_path, color_list, directions, T=32, bbox=None):
    """
    Computes the Euler Characteristic Transform of all the colored-labeled
    barley spikes present in a given directory.
    The ECT of all the seeds in each spike is stored in a CSV.
    One CSV is produced per panicle.

    Attributes:
        `dst` : Destination directory to save the CSV containing
        `meta_path`: Path and filename of the CSV with all the metadata
        `color_list`: List with the color-codes to be expected in the panicle folder
        `directions`: Nx3 np.array with the unit-sized directions to be considered
        `T`: Number of thresholds to be computed for each direction
    """
    scan_name = os.path.normpath(scan_path).split(os.path.sep)[-1]
    x,y,z = 2,1,0
    meta = pd.read_csv(meta_path)

    for color in color_list:
        print('***********************************')
        print(scan_name, color)
        seed_files = glob.glob(scan_path + '*' + color + '*/seed_*_p*.tif')

        if len(seed_files) < 1:
            print('Seeds not found in ', scan_path)

        else:

            csvdst = dst + scan_name + '/'
            if not os.path.isdir(csvdst):
                os.makedirs(csvdst)

            src, fname = os.path.split(seed_files[0])
            label = ((os.path.normpath(src).split(os.path.sep)[-1]).split('_')[0])[-1]

            Tag = [None for i in range(len(seed_files))]
            summary = brew.load_metadata(meta,scan_name,color, len(seed_files), label, [])
            ects = np.zeros((len(seed_files), T*len(directions)), dtype=int)

            for i in range(len(seed_files)):

                raw, fname = os.path.split(seed_files[i])
                Tag[i] = '_'.join(os.path.splitext(fname)[0].split('_')[:3])

                img = tf.imread(seed_files[i])
                img[img > 0] = 1
                img = ndimage.binary_fill_holes(img).astype(img.dtype)
                img = ndimage.binary_closing(img, iterations=2, border_value=0).astype(img.dtype)
                img = ndimage.binary_fill_holes(img).astype(img.dtype)

                cells = brew.complexify(img)
                max_vox = brew.find_tip(cells[0], 2,1,0)
                seed,_,_,_,_ = brew.rotateSVD(cells[0], max_vox)

                ects[i,:] = brew.ECT(seed, cells, directions, T, bbox)

            df = pd.DataFrame(ects)
            summary.index = df.index
            summary['Tag'] = Tag
            summary = pd.merge(summary, df, left_index=True, right_index=True)

            summary.to_csv(csvdst + scan_name + '_' + color +'_d{}_T{}_ect.csv'.format(directions.shape[0],T),
                           index=False)

    return 0


parser = argparse.ArgumentParser(description='Compute the ECT of individual seed scans')

parser.add_argument('dst', metavar='dst_path', type=str,
                    help='path to save results')

parser.add_argument('meta_file', metavar='csv', type=str,
                    help='CSV file with meta data')

parser.add_argument('scan_path', metavar='scan', type=str,
                    help='path to the folder with seed scans')

parser.add_argument('T', metavar='threshold', type=str,
                    help='Number of thresholds')

parser.add_argument('hmin', metavar='h_min', type=str,
                    help='Lower bound height')

parser.add_argument('hmax', metavar='h_max', type=str,
                    help='Upper bound height')

args = parser.parse_args()

#dst = '/home/ejam/documents/barley_stacks/preproc/ects/'
#csv_file = '/home/ejam/documents/barley_stacks/corrected_metadata.csv'
#scan_path = '/home/ejam/documents/barley_stacks/preproc/comps/S123/'
#T = 32

#color_list = ['Blue', 'Orange', 'Red', 'Green']
color_list = ['Red']
#dirs = brew.pole_directions(5, 8, 1,0,2)
dirs = brew.pole_directions(7, 12, 1,0,2)
#dirs = brew.regular_directions(128)
T = int(args.T)
hmin = float(args.hmin)
hmax = float(args.hmax)
bbox = None

if hmin < hmax:
    bbox = (hmin,hmax)

foo = EulerBrew(args.dst, args.meta_file, args.scan_path, color_list, dirs, T, bbox)
