# ###### TRAINING COMMANDS FOR TRIPADVISOR ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--output_dir out/tripadvisor-out-reproduce-profile-title \
--context_in "user profile" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--output_dir out/tripadvisor-out-reproduce-profile-title-and-description \
--context_in "user profile" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--output_dir out/tripadvisor-out-reproduce-review-history-title \
--context_in "review history" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--output_dir out/tripadvisor-out-reproduce-review-history-title-and-description \
--context_in "review history" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--output_dir out/tripadvisor-out-reproduce-item-review-history-title \
--context_in "item-review history" \
--context_out "item title" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--output_dir out/tripadvisor-out-reproduce-item-review-history-title-and-description \
--context_in "item-review history" \
--context_out "item title and description" \
--lr 0.0003 \
--batch_size 64 \
--num_train_epochs 5 \
--seed 42

###### EVALUATION COMMANDS FOR TRIPADVISOR ######
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/tripadvisor-out-reproduce-profile-title \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--context_in "user profile" \
--context_out "item title" \
--output results/tripadvisor-profile-title.jsonl \
--summary_file results/evaluation_summary_comb_tripadvisor.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/tripadvisor-out-reproduce-profile-title-and-description \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--context_in "user profile" \
--context_out "item title and description" \
--output results/tripadvisor-profile-title-and-description.jsonl \
--summary_file results/evaluation_summary_comb_tripadvisor.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/tripadvisor-out-reproduce-review-history-title \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--context_in "review history" \
--context_out "item title" \
--output results/tripadvisor-review-history-title.jsonl \
--summary_file results/evaluation_summary_comb_tripadvisor.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/tripadvisor-out-reproduce-review-history-title-and-description \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--context_in "review history" \
--context_out "item title and description" \
--output results/tripadvisor-review-history-title-and-description.jsonl \
--summary_file results/evaluation_summary_comb_tripadvisor.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/tripadvisor-out-reproduce-item-review-history-title \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--context_in "item-review history" \
--context_out "item title" \
--output results/tripadvisor-item-review-history-title.jsonl \
--summary_file results/evaluation_summary_comb_tripadvisor.json \
--seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate.py \
--pretrained_model out/tripadvisor-out-reproduce-item-review-history-title-and-description \
--dataset TripAdvisor \
--profiles user_profiles/trip_advisor_profiles.json \
--context_in "item-review history" \
--context_out "item title and description" \
--output results/tripadvisor-item-review-history-title-and-description.jsonl \
--summary_file results/evaluation_summary_comb_tripadvisor.json \
--seed 42