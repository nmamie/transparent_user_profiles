###### TRAINING COMMANDS FOR AMAZON SEEDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-seeds37 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 37

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-seeds38 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 38

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-seeds39 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 39

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-seeds40 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 40

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-seeds41 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 41


###### EVALUATION COMMANDS FOR AMAZON SEEDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds37 \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title-seed37.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 37

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds38 \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title-seed38.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 38

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds39 \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title-seed39.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 39

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds40 \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title-seed40.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 40

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds41 \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title-seed41.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 41