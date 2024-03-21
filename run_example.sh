module purge
module load PrgEnv/202310
conda activate /work/lqcd/d20a/users/boyda/.conda/envs/torch2.2
#conda activate /work/lqcd/d20a/users/boyda/.conda/envs/torch2_2023_09_15


export SCRIPTS=/home/lqcd/boyda/codes/dl_run_scripts
export AFFINITY=" --cpu-bind none --mem-bind none $SCRIPTS/polaris_affinity.sh"
export PYTORCH_JIT=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_COLLNET_ENABLE=1
export PYTHONPATH=/home/lqcd/boyda/codes/wandb_to_aim:$PYTHONPATH

date
echo "Start benchmarking"
mpirun  \
 $SCRIPTS/pytorch_launcher.sh \
 $SCRIPTS/print_only_master.sh \
 python bench.py \
   --real_data=False \
   --batch_size=1 \
   --wandb_log=True \
   --wandb_run_name=$NAME-$N_PROCS \
   "$@"

