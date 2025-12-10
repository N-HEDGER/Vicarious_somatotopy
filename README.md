# Vicarious_somatotopy

<img src="https://i.imgur.com/1Zy21Hf.png" alt="Image description" height="200"/>

## Overview

This repository contains the scripts for performing multi-source spectral connective field model fitting as described in this [preprint](https://www.biorxiv.org/content/10.1101/2024.10.21.619382v1). Beyond standard scientific python libraries, the two main packages that drive the analyses are [himalaya](https://gallantlab.org/himalaya/getting_started.html) for model fitting and [pycortex](https://gallantlab.org/pycortex/install.html) for surface ultilities and manipulation.

## ⚙ Installation and Setup.

This software has been tested on Rocky Linux 8.9 (Green Obsidian). Follow these steps to setup:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Vicarious_somatotopy.git
cd Vicarious_somatotopy
```

### 2. You will need to create a python environment that replicates that used to perform the analysis.

```bash
conda create -n testenv python=3.10.2
conda activate testenv
```

### 3. You will then need to install _vicsompy_ - the package associated with this repository.

```bash
pip install -e .
```

This should recognise and install all the dependencies associated with the package, which are defined in the _setup.cfg_ file. Full environment details are also contained in the _environment.yml_ file.

### 4. You will also need to download the pycortex subject ['hcp_999999_draw_NH'](https://drive.google.com/file/d/1IwsSwU9vSt1QQKvHaOxkwO5CPcEtZw88/view?usp=sharing)

and put it in your pycortex directory.

### 5. You will need to download the [source region](https://drive.google.com/file/d/1oHYetfeQ1jQaBtW8xfDi-kaUU38hmXJZ/view?usp=sharing) directory of surfaces, lookup tables and masks for V1 and S1.

### 6. You will need to change the following in the _config/config.yml_ file:

```yaml
paths:
  in_base: "/tank/shared/2019/visual/hcp_{experiment}/" # Where are the HCP data stored?
  out_base: "/tank/hedger/DATA/vicsompy_outputs" # Where do you want the model fits to be output?
  plot_out: "/tank/hedger/scripts/Vicarious_somatotopy/results" # Where do you want the plots to be output?

source_regions:
  source_region_dir: "/tank/hedger/scripts/Sensorium/data" # Where are the source regions stored?
```

### Example data

Example data (HCP average subject) can be found [here](https://drive.google.com/file/d/110OutjDUsOfIbTjJYHBH9Q2P_a0ZmnJw/view?usp=sharing) (broken link now fixed). This directory can be put inside the directory you define in _in_base_ in the yaml file described above. This will then allow you to run the following inside **_notebooks/HCP Fitting_** . Himalaya leverages tqdm and so will include a progress bar to indicate how long to expect for model fitting.

```python
av_fit=analyse_subject('movie','999999',analysis_name='TEST')
```

The expected output is a csv file containing the following columns:

- train*scores*_modality_\_score: Within-set variance explained for modality.
- test*scores*_modality_\_score: Out of-set variance explained for modality.
- best_alphas: ridge alphas for the given voxel.
- spliced*params*_param_\__modality_: The connective field derived quantification for the modality (e.g. _eccentricity_visual_).
- null*score*_modality_: The null (nonspatial) model score for the modality.

### 🕑 Expected installation time

The expected installation time, inclusive of donwloads and package installation should be less than 30 minutes.

## 📒 Notebooks

- The main notebook that drives the main analysis is in: **_notebooks/HCP Fitting_**.📘

- Cortical flatmaps for each of the figures are produced in **_notebooks/Aggregate Plot_** and output to **_results_** folder. 📘

Additional notebooks are in **_notebooks/revised_**.

## 📁 Configuration files ⚙

- The parameters underlying these analyses are in **_config/config.yml_**.

- The parameters driving the plots are in **_config/plot_config.yml_**.

## 🐍 Python scripts

- _vicsompy/subject.py:_ For loading in subject data. 📜

- _viscompy/modeling.py:_ For performing the connective field modeling 📜

- _viscompy/aggregate.py:_ For aggregating outcomes. 📜

- _viscompy/surface.py:_ For handling surface data. 📜

- _viscompy/utils.py:_ Various utilities. 📜

- _viscompy/vis.py:_ For plotting. 📜
