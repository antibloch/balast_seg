#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropiate environment
conda activate torcher


python orthographic_projection/orthoprojector.py --pcd "$1"


rm -rf DiffBIR/dataset && mkdir DiffBIR/dataset
rm -rf DiffBIR/results
cp -r orthoimage_blend.png DiffBIR/dataset



mv ref_width.npy inverse_orthographic_projection
mv ref_height.npy inverse_orthographic_projection
mv pcd_np.npy inverse_orthographic_projection



cd DiffBIR

python -u inference.py --task denoise --upscale 1 --version v2 --sampler spaced --steps 50 --captioner none  --cfg_scale 4.0 --input dataset --output results --device cuda --precision fp32 --cleaner_tiled --cleaner_tile_size 256 --cleaner_tile_stride 128 --vae_encoder_tiled --vae_encoder_tile_size 256 --vae_decoder_tiled --vae_decoder_tile_size 256 --cldm_tiled --cldm_tile_size 512 --cldm_tile_stride 256

mv results/orthoimage_blend.png results/orthoimage_IR.png


cd ..


cp -r DiffBIR/results/orthoimage_IR.png IR_post_process

cd IR_post_process

conda activate torcher
python main3_oracle.py

cd ..

mv IR_post_process/segmented.png inverse_orthographic_projection

cd inverse_orthographic_projection

python inv_orthographicprojector.py

rm -f ref_width.npy
rm -f ref_height.npy
rm -f pcd_np.npy


cd ..

mv inverse_orthographic_projection/final_projected_point_cloud.ply .


python convert_ply_2_laz.py


