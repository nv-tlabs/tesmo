# Data Generation and Augmentation

This folder contains tools to generate diverse scene-aware or object-aware human motion data using existing datasets.

## Locomotion in 3D-FRONT Scene

### Data Sources
#### HumanML3D Dataset
1. Follow instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git)
2. Copy the dataset to our repository:
   ```shell
   cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
   ```
3. Set `$HUMANML_3D_ROOT` to the HumanML3D dataset folder

#### AMASS Dataset
1. Download from [AMASS](https://amass.is.tue.mpg.de/)
2. Set `$AMASS_DATA` to the dataset folder

#### 3D-FRONT Data
1. Download from [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
2. Set `$threeDFront_root` to the dataset folder.
3. Get the bird-view floor plan and object mask for each scene, you can refer to this [scripts](https://github.com/yhw-yhw/MIME/blob/master/scripts/preprocess_data_humanAware.py).


### Fitting Scripts
- Fitting script: `data_generation/locomotion/align_motion_amass.py`

## Human-Object Interaction

Based on [Summon](https://lijiaman.github.io/projects/summon/), we predict contact areas for each motion frame and fit objects of corresponding categories.

### Data Sources

1. SAMP dataset: Download from [SAMP](https://samp.is.tue.mpg.de/) and set `$DATA_ROOT/SAMP`
2. 3D-FUTURE dataset: 
   - Download from [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)
   - Set `$DATA_ROOT/3D-FUTURE-model`
   - We use `raw_model.obj` for each subject


### Fitting Scripts
1. Fit objects to predicted contact areas: `data_generation/interaction/summon/fit_best_obj.py`
2. Calculate transform matrix and merge into `.pkl`: `data_generation/interaction/summon/sort_out_result.py`
3. Visualization: `data_generation/interaction/summon/vis_fitting_results.py`
