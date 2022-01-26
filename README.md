Implementation of SASRec model via pytorch/lightning. 
Originally based on [this code](https://github.com/pmixer/SASRec.pytorch) but rewritten completely.  
[Implementation by authors of paper](https://github.com/kang205/SASRec)


Code for running multiple GPU training:
```
PL_TORCH_DISTRIBUTED_BACKEND=nccl python SASRecMain.py --dataset=ml-1m --maxlen=200 --dropout_rate=0.2 --d_model=50 --num_blocks=2 --num_heads=1 --ndcg_samples=100 --top_k=10 --opt=AdamW --lr=0.001 --weight_decay=1 --batch_size=1024 --num_epochs=300 --use_swa=True --swa_epoch_start=0.65 --swa_annealing_epochs=10 --xavier_init=True --strategy=ddp_spawn --precision=16 --accelerator=auto --devices=auto --l2_pe_reg=1
```
Don't forget to run tensorboard as well
```
tensorboard --logdir ./lightning_logs/ --host 0.0.0.0
```
To use in inference mode run
```
python SASRecMain.py --dataset=ml-1m --inference_only=True --checkpoint_path=./sasrec.ckpt --accelerator=auto
```
This will produce metrics on validation dataset similar to those:
```
DATALOADER:0 VALIDATE RESULTS
{'hr_val': 0.8273178935050964, 'ndcg_val': 0.5920551419258118}
```

To run interactive version use [notebook](./SASRec_interactive.ipynb) 


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
