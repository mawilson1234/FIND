# for whatever reason,
# attempting to remove these files
# at the point in the main script where
# we can do it doesn't work and causes it
# to crash. so, we can run this script after
# all the find scripts have finished running
# to reclaim the disk space by deleting the
# unneeded model checkpoints.

# NOTE THAT YOU SHOULD ONLY RUN THIS AFTER ALL
# THE FIND JOBS HAVE FINISHED RUNNING, OTHERWISE
# IT WILL CAUSE IN-PROGRESS ONES TO FAIL

# Do this using a slurm job dependency
# sbatch --dependency=afterok:$jobid $bash_script_calling_this_script.sh

import os
from glob import glob

def main() -> None:
    # get the model checkpoints
    checkpoints = glob('results/**/*.pt', recursive=True)
    for checkpoint in checkpoints:
        os.remove(checkpoint)

if __name__ == '__main__':
    main()
    
