###### TRAINING COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-profile-title-and-description \
--context_in "user profile" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-review-history-title \
--context_in "review history" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-review-history-title-and-description \
--context_in "review history" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-item-review-history-title \
--context_in "item-review history" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--output_dir amazon-out-reproduce-item-review-history-title-and-description \
--context_in "item-review history" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42