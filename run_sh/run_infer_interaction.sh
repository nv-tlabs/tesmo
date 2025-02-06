# ! ours
batch=70
ckpt=model000030000.pt
guidance_param=2.5
save_postfix=${ckpt}_newobjCFG_threeInference
motion_control_guidance_param=0.0
# --text_condition 'a person sit down and strech out his two legs and cross his arms' \
# --text_condition 'a person sits down cross legged' \
# --text_condition 'a person sit down and strech out his two legs' \
# text="a person sits, both hands on thighs, lifts left hand, bent at the elbow, moving arm up and down."
# text=""
# text="a person sit down and strech out his two legs"
# text="a person sits, both hands on thighs, lifts left hand, bent at the elbow, moving arm up and down."
# text="a person sits, and cross his arms"
text=""
save_text=${text// /_}
save_name=${save_postfix}_batch${batch}_textcfg${guidance_param}_objcfg${motion_control_guidance_param}_${save_text}
if true; then
# ! get the bps feature, and with original pose input. 
ckpt_dir=pretrained_model/interaction
CUDA_VISIBLE_DEVICES=1 python -m sample.infer_motion_control --model_path ${ckpt_dir}/${ckpt} \
    --dataset humanml \
    --opt_name dataset/interaction_controlnet_with_humanml_start_finalpelvis_text_test.txt \
    --arch trans_enc_new_attention \
    --input_feats 268 \
    --batch_size ${batch} --num_samples ${batch} \
    --guidance_param $guidance_param \
    --root_representation root_pos_abs_rotcossin_height_pose_with_humanml \
    --eval \
    --eval_object_collision \
    --device 0 \
    --indicator_double \
    --condition_type text_startOnly_finalPosPelvis_extraToken_CMDMObj_objectBPS \
    --cfg_motion_control \
    --motion_control_guidance_param ${motion_control_guidance_param} \
    --output_dir demo_output/interaction_results/$save_name \
fi




