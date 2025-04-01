#!/bin/bash
# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=18000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=6

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=12:00:00

# =====================
# Logging information
# =====================

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# #SBATCH --partition=PGR-Standard

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=ip
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# copies the collated data for Ed and WMH onto the cluster folder

# copies the collated data for Ed and WMH onto the cluster folder

home_dir=/home/${USER}/ipdis/data/preprocessed_data/collated
ed_home_dir=${home_dir}/EdData
wmh_home_dir=${home_dir}/WMH_challenge_dataset
scratch_dir=/disk/scratch/${USER}/ipdis/preprep/out_data/collated
ed_scratch=${scratch_dir}/EdData
wmh_scratch=${scratch_dir}/WMH_challenge_dataset

mkdir -p ${scratch_dir}

# echo Copying Ed Data
# rsync --archive --update --compress --progress ${ed_home_dir}/ ${ed_scratch}

# # ==============================
# # Finally, run the experiment!
# # ==============================
# # Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# # ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# # you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# # inclusive.

# experiment_text_file=$1
# COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
# echo "Running provided command: ${COMMAND}"
# eval "${COMMAND}"
# echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

result_path=/disk/scratch/${USER}/results/revamped_models
result_dest_path=/home/${USER}/ipdis/results/revamped_models
rsync --archive --update --compress --progress ${result_path}/ ${result_dest_path}

# ================================
# Deleting data from result path
# ================================
# deleting result data
# commented this out because it buggers the logger.
#rm -rf ${result_path}/*

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
