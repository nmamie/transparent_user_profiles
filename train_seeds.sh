###### TRAINING COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 37

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 38

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 39

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 40

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 41


###### EVALUATION COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--seed 37

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--seed 38

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--seed 39

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--seed 40

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-profile-title \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--seed 41