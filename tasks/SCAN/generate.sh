for SPLIT in addprim_jump # addtwicethrice_jump
do
	python generate_data.py --split="${SPLIT}" --shuffle-train
done
