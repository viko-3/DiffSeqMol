# DiffSeqMol
DiffSeqMol: Diffusion model for molecular sequence generation

## Unconditional Genration
If you want to use optimizition code, please switch to the master branch

### train

### sample

## Optimizition
If you want to use optimizition code, please switch to the opt branch

### train
```python
CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=port
  --use_env run_train.py 
  --diff_steps diff_steps 
  --lr learing_rate 
  --learning_steps learning_step
  --save_interval save_interval 
  --seed 102 
  --noise_schedule sqrt 
  --hidden_dim hidden_dim 
  --bsz batch_size 
  --dataset data 
  --data_dir work_dir  
  --vocab bert 
  --seq_len seq_len 
  --schedule_sampler lossaware 
  --notes qqp 
  --config_name seyonec/PubChem10M_SMILES_BPE_450k  
  --microbatch batch_size
```

### sample
```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12356 
--use_env sample_seq2seq.py 
--model_path  model_path
--step diff_steps
--batch_size batch_size 
--seed2 123 
--split valid 
--out_dir generation_outputs 
--top_p -1
```