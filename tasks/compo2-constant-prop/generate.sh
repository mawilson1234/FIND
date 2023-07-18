# Copyright (c) Facebook, Inc. and its affiliates.

for N in 1 6 24 36 ; do
  python generate_data.py --train_N=$N --total_compo_examples=36
done
