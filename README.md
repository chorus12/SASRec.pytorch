Implementation of SASRec model via pytorch/lightning. 
Originally based on [this code](https://github.com/pmixer/SASRec.pytorch) but rewritten completely. Original files preserved (`main.py`, `model.py`, `utils.py`)
[Implementation by authors of paper](https://github.com/kang205/SASRec)


Code for running multiple GPU training:
```
PL_TORCH_DISTRIBUTED_BACKEND=nccl python SASRemMain.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```
Don't forget to run tensorboard as well
```
tensorboard --logdir ./lightning_logs/ --host 0.0.0.0
```
To run interactive version use [notebook](./SASRec.pytorch/SASRec_refactored.ipynb) 


```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```
