# Describing `demeter`: using the Euler characteristic to quantify the shape and biology

**Author**: Erik Amézquita

**Date**: March 2021

## Description

Shape is data and data is shape. Biologists are accustomed to thinking about how the shape of biomolecules, cells, tissues, and organisms arise from the effects of genetics, development, and the environment. Traditionally, biologists use morphometrics to compare and describe shapes. The shape of leaves and fruits is quantified based on homologous landmarks&mdash; similar features due to shared ancestry from a common ancestor&mdash; or harmonic series from a Fourier decomposition of their closed contour. While these methods are useful for comparing many shapes in nature, they can not always be used: there may not be homologous points between samples or a harmonic decomposition of a shape is not appropriate. Topological data analysis (TDA) offers a more comprehensive, versatile way to quantify plant morphology. [TDA](https://doi.org/10.1038/srep01236) uses principles from algebraic topology to comprehensively measure shape in datasets, which reveal morphological features not obvious to the naked eye. In particular, [Euler characteristic curves (ECCs)](https://doi.org/10.1093/imaiai/iau011) serve as a succinct, computationally feasible topological signature that allows downstream statistical analyses. For example, ECCs have been successfully used to to determine the genetic basis of leaf shape in [apple](https://doi.org/10.1104/pp.18.00104), [tomato](https://doi.org/10.1038/s41438-019-0146-2), and [cranberry](https://10.7717/peerj.5461). 

Here we present `demeter`, a python package to quickly compute the ECC of any given grayscale image in linear time with respect to its number of pixels. With `demeter` we provide all the necessary tools to explore the ECC, which can be thought as a two-step procedure. First, we give an image its adequate topological framework, a dual cubical complex in this case. Second, we associate every pixel a number with a fixed real-valued function, known as a filter function. We provide a set of different filter functions that can highlight diverse features in the images, such as Gaussian density, eccentricity, or grayscale intensity. The `demeter` workflow can readily take either 2D pixel-based or 3D voxel-based images, use filter functions outside the ones provided with minimal wrangling, and overall benefit from the standard python ecosystem.

## To install

For now, the package has to be installed from source. You have to 

* download the code with:
```shell
git clone https://github.com/amezqui3/demeter/
```
* move to the directory:
```shell
cd demeter
```
* install with:
```shell
pip install -e .
```

* the `-e` flag is optional. It stands for _editable_, so changes in the source python files will be immediately reflected whenever importing the package.

The package can then be imported in a python shell with:
```shell
import demeter
``` 

## Contents

- `demeter`
    - source python files. Check the details by reading the tutorial and jupyter notebooks
- `doc`
    - documentation. _Work in progress_
- `example_data`
    - sample 2D and 3D TIFF image files
- `jupyter`:
    - `01_density_groundtruth.ipynb`: (python) Select an arbitrary X-ray CT scan an take its density values as ground-truth.
    - `02_normalization.ipynb`: (python) Adjust the rest of raw scans densities to the "ground-truth"
    - `03_cleaning_barley.ipynb`: (python) _ad-hoc_ image processing to remove air/foam/debris from barley panicles and segment out individual spikes
    - `04_labelling_spikes.ipynb`: (python) Label individual spikes based on their relative position in the original scan
    - `05_seed_isolation.ipynb`: (python) Segment out individual seeds from every spike
    - `06_traditional_computation.ipynb`: (python) For each seed, measure its shape with traditional shape descriptors such as length, width, or volume.
    - `07_complexify_binary_image.ipynb`: (python) Details on how to get a 2D dual cubical complex from a grayscale image
    - `08_sphere_directions.ipynb`: (python) Details on how to define 3D directions either uniformly or randomly placed.
    - `09_ect_computation.ipynb`: (python) Details on how to compute ECCs and the ECT of a grayscale image
    - `10_shape_descriptor_classification.ipynb`: (R) Train an SVM to classify 28 different barley lines based solely on the shape of their grains.
    - `12_shape_descriptor_analysis.ipynb`: (R) Compute and plot classification accuracy of traditional vs topological shape descriptors
- `outputs`: 
    - Directory where toy outputs from the jupyter notebooks are written
- `scripts`:
    - Python scripts rather than notebooks to perform all the data processing and analysis routines. _Work in progress_.
- `tutorials`:
    - `aatrn2021_euler_characteristic_transform.ipynb` (python) A gentle introduction to the ECT. Written originally for the [AATRN Tutorial-a-thon](https://sites.google.com/view/aatrn-tutorial-a-thon)
    - `glbio2021_quantify_the_shape_in_biology.ipynb` (python) A gentle introduction to the ECT and `demeter`. Written originally for the [GLBIO 2021 conference](https://www.iscb.org/glbio2021).
    - `nappn2021_shape_of_things_to_come.ipynb` (python) A gentle introduction to TDA and the ECT. Written originally for the [2021 Annual Meeting of the NAPPN](https://www.nappn2021.org).

## To run locally

If you want to run locally all the notebooks, make sure you have `jupyter` enabled for **both** python and R.

You will also need to have the following python libraries

     matplotlib scipy numpy tifffile pandas

And the following R libraries

     ggplot2 ggdendro e1071 reshape2 dplyr viridis kernlab PMCMRplus
