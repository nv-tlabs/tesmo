fitting_folder='dataset/3dfront_fitting/align_data_obj_v2' # change this folder to your fitting folder
scene_folder='dataset/3dfront_scene_mask/livingroom' # change this fodler to your folder which stores the bird-view floor plan and object mask
split='train'  # 'train' or 'test'

python scripts/preprocess_canonical_motion_scene_pairs.py --split ${split}  \
    --fitting_folder ${fitting_folder}  \
    --scene_folder ${scene_folder}