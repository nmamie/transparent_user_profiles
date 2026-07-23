# ###### TRAINING COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-profile-title-and-description \
--context_in "user profile" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-review-history-title \
--context_in "review history" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-review-history-title-and-description \
--context_in "review history" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-item-review-history-title \
--context_in "item-review history" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--output_dir out/amazon-out-reproduce-item-review-history-title-and-description \
--context_in "item-review history" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

###### EVALUATION COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--summary_file results/evaluation_summary_comb.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title-and-description \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title and description" \
--output results/profile-title-and-description.jsonl \
--summary_file results/evaluation_summary_comb.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-review-history-title \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "review history" \
--context_out "item title" \
--output results/review-history-title.jsonl \
--summary_file results/evaluation_summary_comb.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-review-history-title-and-description \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "review history" \
--context_out "item title and description" \
--output results/review-history-title-and-description.jsonl \
--summary_file results/evaluation_summary_comb.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-item-review-history-title \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "item-review history" \
--context_out "item title" \
--output results/item-review-history-title.jsonl \
--summary_file results/evaluation_summary_comb.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-item-review-history-title-and-description \
--dataset Amazon/MoviesAndTV \
--profiles user_profiles/amazon_profiles.json \
--context_in "item-review history" \
--context_out "item title and description" \
--output results/item-review-history-title-and-description.jsonl \
--summary_file results/evaluation_summary_comb.json \
--seed 42