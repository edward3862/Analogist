# colorization
python evaluation.py \
    --input_dir datasets/low_level_tasks_processed/colorization_masked \
    --prompt_file datasets/low_level_tasks_processed/colorization_gpt4_out.txt \
    --output_dir results/low_level_tasks/colorization \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15

# deblur
python evaluation.py \
    --input_dir datasets/low_level_tasks_processed/deblur_masked \
    --prompt_file datasets/low_level_tasks_processed/deblur_gpt4_out.txt \
    --output_dir results/low_level_tasks/deblur \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.7 --scale_sac 1.3 --guidance_scale 15

# denoise
python evaluation.py \
    --input_dir datasets/low_level_tasks_processed/denoise_masked \
    --prompt_file datasets/low_level_tasks_processed/denoise_gpt4_out.txt \
    --output_dir results/low_level_tasks/denoise \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.7 --scale_sac 1.3 --guidance_scale 15

# enhancement
python evaluation.py \
    --input_dir datasets/low_level_tasks_processed/enhancement_masked \
    --prompt_file datasets/low_level_tasks_processed/enhancement_gpt4_out.txt \
    --output_dir results/low_level_tasks/enhancement \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15

# image editing
python evaluation.py \
    --input_dir datasets/manipulation_tasks_processed/image_editing_masked \
    --prompt_file datasets/manipulation_tasks_processed/image_editing_gpt4_out.txt \
    --output_dir results/manipulation_tasks/image_editing \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15

# image translation
python evaluation.py \
    --input_dir datasets/manipulation_tasks_processed/image_translation_masked \
    --prompt_file datasets/manipulation_tasks_processed/image_translation_gpt4_out.txt \
    --output_dir results/manipulation_tasks/image_translation \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15

# style transfer
python evaluation.py \
    --input_dir datasets/manipulation_tasks_processed/style_transfer_masked \
    --prompt_file datasets/manipulation_tasks_processed/style_transfer_gpt4_out.txt \
    --output_dir results/manipulation_tasks/style_transfer \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15

# inpainting
python evaluation.py \
    --input_dir datasets/vision_tasks_processed/inpainting_masked \
    --prompt_file datasets/vision_tasks_processed/inpainting_gpt4_out.txt \
    --output_dir results/vision_tasks/inpainting \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 7.5

# ubcfashion
python evaluation.py \
    --input_dir datasets/vision_tasks_processed/ubcfashion_masked \
    --prompt_file datasets/vision_tasks_processed/ubcfashion_gpt4_out.txt \
    --output_dir results/vision_tasks/ubcfashion \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15

# scannet
python evaluation.py \
    --input_dir datasets/vision_tasks_processed/scannet_masked \
    --prompt_file datasets/vision_tasks_processed/scannet_gpt4_out.txt \
    --output_dir results/vision_tasks/scannet \
    --num_images_per_prompt 8 --res 512 \
    --sac_start_layer 3 --sac_end_layer 11 \
    --cam_start_layer 3 --cam_end_layer 11 \
    --strength 0.98 --scale_sac 1.3 --guidance_scale 15