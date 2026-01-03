#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=scan_decontam
#SBATCH --account=AIFAC_L01_028
#SBATCH -p boost_usr_prod
#SBATCH --threads-per-core=1
#SBATCH --time=23:59:00
#SBATCH --output=/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/decontaminate/slurm-out/%j-scan.out

DATA_PATH="/leonardo_work/AIFAC_L01_028/shared/datasets/raw/ontocord-MixtureVitae-211BT-parquet/*.parquet"
# python decontam_hf.py scan --input-glob "$DATA_PATH" \
#                             --index "index.native.pkl" \
#                             --text-key "text" \
#                             --id-key "id" \
#                             --out-dir "decontam_results" \
#                             --workers 64 \
#                             --min-hits 3

python decontam_hf.py filter  --input-glob "$DATA_PATH" \
                            --contaminated-dir "/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/decontaminate/decontam_results/contaminated_docs" \
                            --out-dir "/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/cleaned_dataset"