#!/bin/bash
#
#SBATCH --job-name=merge_meteo
#SBATCH --output=output/merge/output_%A_%a.txt
#SBATCH --error=output/merge/error_%A_%a.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Get line from list (only crop-country, ignore indicator)
line=$(awk "NR==${SLURM_ARRAY_TASK_ID}+1" /lustre/backup/SHARED/AIN/agml/crop_country_list_simple.txt)
crop=$(echo "$line" | cut -d'_' -f1)
country=$(echo "$line" | cut -d'_' -f2)

# Change to working directory
cd /lustre/backup/SHARED/AIN/agml/AgML-CY-Bench/

# Run merge script
echo "Merging region: $crop $country"
poetry run python /lustre/backup/SHARED/AIN/agml/predictors/merge_agera5.py "$crop" "$country"