## Generalized-activated Deep Double Deterministic Policy Gradient

`pureDDPG` is the vanilla DDPG and `DDPG` is the fine-tuned version of `pureDDPG` which could achieve much better performance than `DDPG` at various environments. TD3 use fine-tuned `DDPG` as the baselines, and so as in our work.

### Run GD3
```python
CUDA_VISIBLE_DEVICES="1" python main.py --env 'Reacher-v2' --seed 1 --policy 'GD3' --activate 'poly' --beta 2 --dir './logs/GD3/poly2/Reacher/r0' --save-model
```

### Run TD3
```python
CUDA_VISIBLE_DEVICES="1" python main.py --env 'Ant-v2' --seed 1 --policy 'TD3' --dir './logs/TD3/Ant/r0' --save-model
```

### Run DDPG
```python
CUDA_VISIBLE_DEVICES="1" python main.py --env 'Ant-v2' --seed 1 --policy 'DDPG' --dir './logs/DDPG/Ant/r0' --save-model
```

### Run MGD3 (Mixed GD3)
MGD3 refers to the mix of different activation function at different stages of training. There are generally two methods of blending different activation functions:

- use one activation function at the first stage of training and change it to another one afterwards
- use a decay parameter to combine two activation functions such that with training timesteps increases, the estimated value would lean to the other activation gradually

```python
CUDA_VISIBLE_DEVICES="1" python main.py --env 'HalfCheetah-v2' --seed 1 --policy 'MGD3' --first-activate 'poly' --first-beta 2 --second-activate 'softmax' --dir './logs/MGD3/twophase/poly2+softmax/HalfCheetah/r0' --save-model
```
