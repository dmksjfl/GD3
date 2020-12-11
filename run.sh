#!/bin/bash

for ((i=0;i<5;i+=1))
do 
	#python main.py --env 'Ant-v2' --seed $i 
	#python main.py --env 'BipedalWalker-v2' --seed $i
	python main.py --env 'HalfCheetah-v2' --seed $i --policy 'TD3' --dir './logs/TD3/' --save-model
	python main.py --env 'HalfCheetah-v2' --seed $i --policy 'SD3' --dir './logs/SD3/' --save-model
	python main.py --env 'HalfCheetah-v2' --seed $i --policy 'GD3' --activate 'poly' --beta 2 --dir './logs/GD3/' --save-model
	#python main.py --env 'Hopper-v2' --seed $i
	#python main.py --env 'LunarLanderContinuous-v2' --seed $i
	#python main.py --env 'Swimmer-v2' --seed $i
	#python main.py --env 'Walker2d-v2' --seed $i
	#python main.py --env 'Humanoid-v2' --hidden-sizes 256,256 --batch-size 256 --actor-lr 3e-4 --critic-lr 3e-4 --seed $i
done