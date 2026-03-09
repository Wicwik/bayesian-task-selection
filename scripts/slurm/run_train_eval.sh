#SBATCH --partition=GPU
#SBATCH --account=perun250162
#SBATCH --qos=perun250162
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=48G

module load libsndfile

export HF_HOME="/lustre/scratch/$USER/huggingface"

llamafactory-cli train $1
llamafactory-cli train $2