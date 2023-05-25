# DiffSeqMol
DiffSeqMol: Diffusion model for molecular sequence generation

## Unconditional Genration
If you want to use optimizition code, please switch to the master branch

### train

### sample

## Optimizition
If you want to use optimizition code, please switch to the opt branch

### train
’‘’
CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12228  --use_env run_train.py --diff_steps 500 --lr 0.0001 --learning_steps 5000000 --save_interval 500000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 256 --dataset qqp --data_dir /work/data1/cyy/DataSet/language_model/md   --vocab bert --seq_len 200 --schedule_sampler lossaware --notes qqp --config_name seyonec/PubChem10M_SMILES_BPE_450k  --microbatch 256
‘’‘
### sample