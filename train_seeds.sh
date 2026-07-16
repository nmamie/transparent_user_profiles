###### TRAINING COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-seeds37 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 37

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-seeds38 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 38

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-seeds39 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 39

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-seeds40 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 40

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-seeds41 \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 41


###### EVALUATION COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds37 \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 37

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds38 \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 38

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds39 \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 39

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds40 \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 40

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/amazon-out-reproduce-seeds41 \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--summary_file results/evaluation_summary_seeds.json \
--seed 41