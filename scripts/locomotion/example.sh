#!/bin/bash

# # Generate synthetic data
# python scripts/locomotion/make_data.py \
# 	--base_datadir ./logs/data \
# 	--env_name hopper-medium-v2 \
# 	--diffusers_repo leekwoon/hopper-medium-v2-H32-T20 \
# 	--n 500000

# # Compute restoration gap
# python scripts/locomotion/compute_restoration_gaps.py \
# 	--env_name hopper-medium-v2 \
# 	--data_path ./logs/data/hopper-medium-v2-H32-T20/500000_finish.npz \
# 	--diffusers_repo leekwoon/hopper-medium-v2-H32-T20 \
# 	--strength 0.9 \
# 	--num_plan 64

# # Train gap predictor
# python scripts/train_gap_predictor.py \
# 	--base_logdir ./logs/gap_predictor \
# 	--env_name hopper-medium-v2 \
# 	--data_path ./logs/data/hopper-medium-v2-H32-T20/500000_finish.npz \
# 	--score_path ./logs/data/hopper-medium-v2-H32-T20/500000_finish_restoration_gaps.npy \
# 	--seed 0

# Evaluate RGG (with pretrained gap predictor)
python scripts/locomotion/evaluate_rgg.py \
	--logbase ./logs/evaluate \
	--env_name hopper-medium-v2 \
	--diffusers_repo leekwoon/hopper-medium-v2-H32-T20 \
	--num_episodes 15 \
	--spec rgg \
	--gap_predictor_path ./logs/gap_predictor/hopper-medium-v2-H32-T20/500000_finish/2023_04_06_22_37_41/seed_0/state_best.pt
    
# Evaluate RGG+
python scripts/locomotion/evaluate_rggplus.py \
	--logbase ./logs/evaluate \
	--env_name hopper-medium-v2 \
	--diffusers_repo leekwoon/hopper-medium-v2-H32-T20 \
	--num_episodes 15 \
	--spec rggplus \
	--gap_predictor_path ./logs/gap_predictor/hopper-medium-v2-H32-T20/500000_finish/2023_04_06_22_37_41/seed_0/state_best.pt
    
