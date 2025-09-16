#!/bin/bash
#SBATCH --job-name=agera5_download
#SBATCH --output=output/agera5/output_%A_%a_%j.txt
#SBATCH --error=output/agera5/error_%A_%a_%j.txt
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --array=0-160%10  # This will cover all year×param combinations

# Initialize modules
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Change to working directory
cd /lustre/backup/SHARED/AIN/agml/AgML-CY-Bench/

# Define years and parameters
YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023)
PARAMS=("Maximum_Temperature" "Minimum_Temperature" "Mean_Temperature" "Solar_Radiation_Flux" "Precipitation_Flux" "Reference_Evapotranspiration" "Vapour_Pressure_Deficit")

# Map SLURM_ARRAY_TASK_ID to year and param
TOTAL_PARAMS=${#PARAMS[@]}
YEAR_INDEX=$(( SLURM_ARRAY_TASK_ID / TOTAL_PARAMS ))
PARAM_INDEX=$(( SLURM_ARRAY_TASK_ID % TOTAL_PARAMS ))

YEAR=${YEARS[$YEAR_INDEX]}
PARAM=${PARAMS[$PARAM_INDEX]}

echo "Downloading $PARAM $YEAR"

# Run the Python script
poetry run python /lustre/backup/SHARED/AIN/agml/predictors/AgERA5/download_agera5.py --year $YEAR --param $PARAM

