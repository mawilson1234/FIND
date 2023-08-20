for SPLIT in simple_original # addprim_jump_original addprim_jump addtwicethrice_jump
do
	python generate_data.py --split="${SPLIT}" # --shuffle-train
done
