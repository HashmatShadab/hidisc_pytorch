### SSL Training

```python
# Single GPU

python main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=none
training.batch_size=10 root="./" out_dir="Results"

# Multi GPU

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=none
training.batch_size=10 root="./" out_dir="Results"
```

### SSL Training with Attack

```python
# Single GPU

python main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=pgd training.attack_eps=8.0
training.attack.alpha=2.0 training.attack.iters=5 training.batch_size=10 root="./" out_dir="AdvResults"

# Multi GPU

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=pgd training.attack_eps=8.0
training.attack.alpha=2.0 training.attack.iters=5 training.batch_size=10 root="./" out_dir="AdvResults"
```
