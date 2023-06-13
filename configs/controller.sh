exp=${1:-'test'}
gpu=${2:-'1'}
type=${3:-'local'} # choose slurm if you are running on a cluster with slurm scheduler

if [ "$type" == 'local' ]; then
  extra_args=${@:4:99}
else
  quotatype=${4:-'auto'} # for slurm
  partition=${5:-'1'} # for slurm
  extra_args=${@:6:99}
  quotatype=spot
  partition=YOUR_PARTITION
  extra_args=${@:4:99}
fi

name=${name/#configs/logs}
name=${name//.sh//$exp}
work_dir="${name}"
now=$(date +"%Y%m%d_%H%M%S")
mkdir  -p $work_dir

ncpu='4'

if [ "$quotatype" == 'reserved_normal' ]; then
  quotatype='reserved --phx-priority=${gpu} normal'
fi

if [ "$type" == 'local' ]; then


  ava_path=/mnt/afs/xswu/datasets/AVA/images
  local_data_path=/mnt/afs/xswu/datasets/preference
  local_ava_path=/mnt/afs/xswu/datasets/AVA
  local_simulacra_path=/mnt/afs/xswu/datasets/simulacra
  local_region_path=/mnt/afs/xswu/datasets/regional_dataset
  local_ranking_path=/mnt/afs/xswu/datasets/HPDv2
  local_benchmark_path=/mnt/afs/xswu/datasets/benchmark
  local_ImageReward_path=/mnt/afs/xswu/datasets/ImageReward
  local_pap_path=/mnt/afs/xswu/datasets/PAP

  header="torchrun --nproc_per_node=${gpu} --nnodes=1 --max_restarts=3 -m src.training.main "

else

  data_path=s3://preference_images/
  ava_path=s3://AVA/
  simulacra_path=s3://simulacra/
  region_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/regional_dataset/
  local_data_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/human_preference
  local_ava_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/AVA
  local_simulacra_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/simulacra
  local_region_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/regional_dataset
  local_ranking_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/ranking_dataset
  local_benchmark_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/benchmark
  local_ImageReward_path=/mnt/lustre/wuxiaoshi1.vendor/datasets/ImageReward
  header="srun --async --partition=$partition -n${gpu} --mpi=pmi2 --gres=gpu:$gpu --ntasks-per-node=${gpu} --quotatype=$quotatype \
    --job-name=$exp --cpus-per-task=$ncpu --kill-on-bad-exit=1 -o local.out python -m src.training.main "

fi
