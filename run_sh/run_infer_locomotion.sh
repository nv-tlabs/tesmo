#!/bin/bash
CODE_ROOT_DIR=$(dirname "$0")
echo $CODE_ROOT_DIR


batch=10
guidance_param=2.5
output_dir=demo_output/locomotion_results/cmdm_270ckpt_${batch}_cfg${guidance_param}_wo_blend
python sample/infer_motion_control.py \
    --model_path pretrained_model/locomotion/model000270000.pt \
    --dataset humanml --opt_name dataset/humanml_opt_pos_abs_rotcossin_height_norm_double_indicator_cmdm_hardhalf.txt \
    --arch trans_enc_new_attention --root_representation root_pos_abs_rotcossin_height_only_norm \
    --input_feats 5 --num_samples $batch --guidance_param ${guidance_param} \
    --device 1 \
    --batch_size $batch --condition_type text_startEnd_sceneGrid_cmdm --split test \
    --indicator_double \
    --eval \
    --output_dir $output_dir
    # --show_input \
    # --not_show_video 

### run inpainting with PriorMDM, please download the horizontal controlled model weight from https://drive.google.com/file/d/1xLNza6S8Iz2MqSlMJnL38FPqTQhGnqfY/view?usp=share_link

python -m sample.finetuned_inpainting_input \
 --model_path pretrained_model/inpainting/root_horizontal_finetuned/model000280000.pt \
 --opt_name dataset/humanml_opt.txt \
 --batch_size 10 --num_samples 10 \
 --input_trajectory $output_dir/results.npy \
 --input_motion_kind 'absolute_xz_rot' --input_feats 263 \
 --output_dir $output_dir/priormdm_results \
 --split test --inpainting_mask root_horizontal \
 --text_condition "" \
 --device 1 \
 --joint2mesh


