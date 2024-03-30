# learning_nerf_cotracker
This Project is Cotracker implemented in Learning Nerf Framework


### Training

```
python train_net.py --cfg_file configs/cotracker/cotracker.yaml
```

### Evaluation

```
python run.py --type evaluate --cfg_file configs/cotracker/cotracker.yaml
```

```
python run.py --type visualize --cfg_file configs/cotracker/cotracker.yaml
```

### 查看loss曲线

```
tensorboard --logdir=data/record --bind_all
```
