###### EVALUATION COMMANDS ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model ./amazon-out-reproduce-profile-title/results \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/profile-title.jsonl \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model ./amazon-out-reproduce-profile-title-and-description/results \
--profiles user_profiles/amazon_profiles.json \
--context_in "user profile" \
--context_out "item title and description" \
--output results/profile-title-and-description.jsonl \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model ./amazon-out-reproduce-review-history-title/results \
--profiles user_profiles/amazon_profiles.json \
--context_in "review history" \
--context_out "item title" \
--output results/review-history-title.jsonl \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model ./amazon-out-reproduce-review-history-title-and-description/results \
--profiles user_profiles/amazon_profiles.json \
--context_in "review history" \
--context_out "item title and description" \
--output results/review-history-title-and-description.jsonl \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model ./amazon-out-reproduce-item-review-history-title/results \
--profiles user_profiles/amazon_profiles.json \
--context_in "item-review history" \
--context_out "item title" \
--output results/item-review-history-title.jsonl \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model ./amazon-out-reproduce-item-review-history-title-and-description/results \
--profiles user_profiles/amazon_profiles.json \
--context_in "item-review history" \
--context_out "item title and description" \
--output results/item-review-history-title-and-description.jsonl \
--seed 42