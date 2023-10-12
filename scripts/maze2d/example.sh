#!/bin/bash

# # Generate synthetic data
# python scripts/maze2d/make_data.py \
# 	--base_datadir ./logs/data \
# 	--env_name maze2d-large-v1 \
# 	--diffusers_repo leekwoon/maze2d-large-v1-H384-T256 \
# 	--n 500000

# # Compute restoration gap
# python scripts/maze2d/compute_restoration_gaps.py \
# 	--env_name maze2d-large-v1 \
# 	--data_path ./logs/data/maze2d-large-v1-H384-T256/500000_finish.npz \
# 	--diffusers_repo leekwoon/maze2d-large-v1-H384-T256 \
# 	--strength 0.9 \
# 	--num_plan 10

# # Train gap predictor
# python scripts/train_gap_predictor.py \
# 	--base_logdir ./logs/gap_predictor \
# 	--env_name maze2d-large-v1 \
# 	--data_path ./logs/data/maze2d-large-v1-H384-T256/500000_finish.npz \
# 	--score_path ./logs/data/maze2d-large-v1-H384-T256/500000_finish_restoration_gaps.npy \
# 	--seed 0

# Evaluate RGG (with pretrained gap predictor)
python scripts/maze2d/evaluate_rgg.py \
	--logbase ./logs/evaluate \
	--env_name maze2d-large-v1 \
	--task single_task \
	--diffusers_repo leekwoon/maze2d-large-v1-H384-T256 \
	--num_episodes 1000 \
	--spec rgg \
	--gap_predictor_path ./logs/gap_predictor/maze2d-large-v1-H384-T256/500000_finish/2023_03_26_03_44_06/seed_0/state_best.pt
    
# Evaluate RGG+
python scripts/maze2d/evaluate_rggplus.py \
	--logbase ./logs/evaluate \
	--env_name maze2d-large-v1 \
	--task single_task \
	--diffusers_repo leekwoon/maze2d-large-v1-H384-T256 \
	--num_episodes 1000 \
	--spec rggplus \
	--gap_predictor_path ./logs/gap_predictor/maze2d-large-v1-H384-T256/500000_finish/2023_03_26_03_44_06/seed_0/state_best.pt
    

