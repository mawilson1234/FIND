# Copyright (c) Facebook, Inc. and its affiliates.

for d in 4; do
  python generate_data.py --test_depth=$d --train_n_examples_per_depth=2
done
