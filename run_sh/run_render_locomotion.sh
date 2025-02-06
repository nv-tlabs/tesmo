# ！ visualize 
# ! y-is up-size down.
batch=10
guidance_param=2.5
root_dir=demo_output/locomotion_results/cmdm_270ckpt_${batch}_cfg${guidance_param}_wo_blend
sub_priormdm_dir=priormdm_results
save_dir=${root_dir}/priormdm_results_render
sample_id=$1
python visualize/render_visualize_locomotion.py --input_pkl HUMANISE/align_data_obj_v2_test_store/all_data_test_npz_${batch}_hardHalf_1.pkl \
    --align_dir HUMANISE/align_data_obj_v2_test \
    --root_dir ${root_dir} \
    --sub_priormdm_dir ${sub_priormdm_dir} \
    --sample_id ${sample_id} \
    --save_dir ${save_dir}



