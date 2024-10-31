# Vicarious_somatotopy

Code for performing multi-modal connective field modeling.


<img src="https://i.imgur.com/1Zy21Hf.png" alt="Image description" height="300"/>


### Overview
The main notebook that drives the analysis is in: ***notebooks/HCP Fitting***.

Cortical flatmaps for each of the figures are produced in ***notebooks/Aggregate Plot*** and output to ***results*** folder.

The parameters underlying these analyses are in ***config/config.yml***.

The parameters driving the plots are in ***config/plot_config.yml***.


### Setup.

You will need to install the environment defined in the yml file.


You will need access to the pycortex subject ['hcp_999999_draw_NH'](https://drive.google.com/file/d/1IwsSwU9vSt1QQKvHaOxkwO5CPcEtZw88/view?usp=sharing)
 and put it in your pycortex directory.

You will need to download the [source region](https://drive.google.com/file/d/1oHYetfeQ1jQaBtW8xfDi-kaUU38hmXJZ/view?usp=sharing) directory of surfaces, lookup tables and masks for V1 and S1.

You will need to change the following in the yaml file:


```yaml
paths:
    in_base: "/tank/shared/2019/visual/hcp_{experiment}/" # Where are the HCP data stored?
    out_base: "/tank/hedger/DATA/STRIATUM_prf" # Where do you want the model fits to be output?
    plot_out: "/tank/hedger/scripts/Vicarious_somatotopy/results" # Where do you want the plots to be output?

source_regions:
    source_region_dir: "/tank/hedger/scripts/Sensorium/data" # Where are the source regions stored?

```

### Files

*vicsompy/subject.py:* For loading in subject data.

*viscompy/modeling.py:* For performing the connective field modeling

*viscompy/aggregate.py:* For aggregating outcomes.

*viscompy/surface.py:* For handling surface data.

*viscompy/utils.py:* Various utilities.

*viscompy/vis.py:* For plotting.
