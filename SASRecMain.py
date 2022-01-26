"""
Implementation of Self-attentive sequential recommendation paper:
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
Originally taken [this code](https://github.com/pmixer/SASRec.pytorchhttps://github.com/pmixer/SASRec.pytorch) and rewritten model class plus used lightning.  

on multiple GPU run with command:
PL_TORCH_DISTRIBUTED_BACKEND=nccl python SASRecMain.py --dataset=ml-1m --maxlen=200 --dropout_rate=0.2 --d_model=50 --num_blocks=2 --num_heads=1 --ndcg_samples=100 --top_k=10 --opt=AdamW --lr=0.001 --weight_decay=1 --batch_size=1024 --num_epochs=300 --use_swa=True --swa_epoch_start=0.65 --swa_annealing_epochs=10 --xavier_init=True --strategy=ddp_spawn --precision=16 --accelerator=auto --devices=auto --l2_pe_reg=1

to calc validation metrics run with:
python SASRecMain.py --dataset=ml-1m --inference_only=True --checkpoint_path=./sasrec.ckpt --accelerator=auto


don't forget to launch tensorboard with:
tensorboard --logdir ./lightning_logs/ --host 0.0.0.0

Author: Sergei Bazhin
Date: 2021-DEC - JAN-2022
"""
import os
import numpy as np
import torch
import pytorch_lightning as pl
import argparse
# module with datasets definition = train, validation and test
import DataHelper as DH
import SASRecModel as SASRec
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from torch.nn import MultiheadAttention, LayerNorm, Dropout, Conv1d, Embedding, BCEWithLogitsLoss
from SASRecModel import PointWiseFF, SASRecEncoderLayer, PositinalEncoder, SASRecEncoder


# setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', 
                    required=True, 
                    help="dataset to use : Beauty, ml-1m(default), Steam or Video")

parser.add_argument('--maxlen', default=50, type=int, 
                    help="truncate input sequence to last maxlen items, default 50")
parser.add_argument('--hidden_units', default=50, type=int, help="synonym for d_model") # synonym for d_model
parser.add_argument('--d_model', default=50, type=int, 
                    help="Transformer internal dimention") # same as hidden_units   
parser.add_argument('--num_blocks', default=2, type=int, help="Number of blocks in Transformer")
parser.add_argument('--num_heads', default=1, type=int, help="Number of heads in self-attention")
parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout rate for Transformer")
parser.add_argument('--l2_pe_reg', default=0.1, type=float, help="Regularization for positional embedding")


parser.add_argument('--ndcg_samples', default=100, type=int, 
                    help="How many random items to pick up in hit-rate and ndcg calculation, default 100")
parser.add_argument('--top_k', default=10, type=int, 
                    help="How many items with high scores to pick for hit-rate and ndcg calculation, default 10")
parser.add_argument('--opt', default='Adam', type=str, help="Oplimizer to use: Adam(default), AdmaW, FusedAdam(requires apex library)")
parser.add_argument('--lr', default=0.001, type=float, 
                    help="learning rate, default 0.001")
parser.add_argument('--weight_decay', default=0.001, type=float, help="Weight decay for AdmaW")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--warmup_proportion', default=0.2, type=float, help="Fraction of total optimization steps to increase learning rate from zero to max value")
# for different optimizers - regular Adam uses num_epochs and LAMB uses max_iters
parser.add_argument('--max_iters', default=10000, type=int, help="Optimization budget in update iterations")
parser.add_argument('--num_epochs', default=201, type=int, help="Number of epochs to train")
# swa parameters
parser.add_argument('--use_swa', default=False, type=bool, help="Use Stochastic Weights Ageraging algorythm")
parser.add_argument('--swa_epoch_start', default=0.8, type=float, help="Start SWA after that part of total epochs")
parser.add_argument('--swa_annealing_epochs', default=10, type=int, help="Number of epochs in the annealing phase of SWA")

# xavier init
parser.add_argument('--xavier_init', default=True, type=bool, help="Use xavier normal to init the model")

parser.add_argument('--inference_only', default=False, type=bool)
parser.add_argument('--checkpoint_path', default=None, type=str, help="Path to lightning checkpoint file")

# Torch Lightning settings
# https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
# Data Parallel (strategy='dp') (multiple-gpus, 1 machine)
# DistributedDataParallel (strategy='ddp') (multiple-gpus across many machines (python script based)).
# DistributedDataParallel (strategy='ddp_spawn') (multiple-gpus across many machines (spawn based)).
# DistributedDataParallel 2 (strategy='ddp2') (DP in a machine, DDP across machines).
# Horovod (strategy='horovod') (multi-machine, multi-gpu, configured at runtime)
# TPUs (tpu_cores=8|x) (tpu or TPU pod)
parser.add_argument('--strategy', default='ddp_spawn', type=str, help="Lightning parallel training strategy dp, ddp, ddp_spawn(default), ddp2, etc ")
parser.add_argument('--precision', default=16, type=int, help="Lightning precision for model data during trining 16(default) or 32")
parser.add_argument('--accelerator', default="auto", type=str, help="Lightning accelerator auto(defaut), cpu, gpu, tpu")
parser.add_argument('--devices', default="auto", type=str, 
                    help="Lightning devices to use - see https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#devices")

# args = parser.parse_args(['--dataset=ml-1m', '--train_dir=default',
#                           '--maxlen=200', '--dropout_rate=0.2', '--device=cuda'])

args = parser.parse_args()
args = vars(args)


if __name__ == '__main__':
    # read dataset
    dataset = DH.data_partition(args['dataset'])

    print('\nRuntime parameters\n',*[(k, v) for (k, v) in args.items()], sep="\n")

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # batches got sliced by users, i.e. batch accumulate BATCH_SIZE user sequences of items selected/bought
    BATCH_SIZE = args['batch_size']
    num_batch = len(user_train) // BATCH_SIZE  # number of batches

    user_train_lens = list(map(len, [v for k, v in user_train.items()]))
    print(
        f'average sequence length: {sum(user_train_lens)/len(user_train):.1f}')

    print(f"\nBatch size is - {BATCH_SIZE}\n")


    
    callbacks_list = []
    # save checkpoints
    callbacks_list.append(ModelCheckpoint(monitor="ndcg_val", mode="max", 
                                           filename="sasrec_{epoch:05d}_{step}_{ndgc_val:.4f}"))
    
    # use SWA
    if args['use_swa']:
        callbacks_list.append(StochasticWeightAveraging(swa_epoch_start=args['swa_epoch_start'], 
                                                        annealing_epochs=args['swa_annealing_epochs']))
                             
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = pl.Trainer(strategy=args['strategy'],
                         accelerator=args['accelerator'],
                         devices=args['devices'],
                         max_epochs=args['num_epochs'],
                         reload_dataloaders_every_n_epochs=1,
                         val_check_interval=1.0,
                         callbacks=callbacks_list,
                         # log 4 times per epoch
                         log_every_n_steps=int(
                             len(user_train) / args['batch_size'] / 3),
                         # log_every_n_steps=1,
                         # limit_val_batches=0, How much of validation dataset to check. Useful when debugging or testing something that happens at the end of an epoch.
                         num_sanity_val_steps=1)    
    
    # no training but only validation metrics
    if args['inference_only']:
        model = SASRecEncoder.load_from_checkpoint(args['checkpoint_path'])
        val_loader = torch.utils.data.DataLoader(dataset=DH.SequenceDataValidation(user_train, 
                                                                           user_valid, 
                                                                           usernum, 
                                                                           itemnum, 
                                                                           model.hparams.maxlen, 
                                                                           model.hparams.ndcg_samples),
                                                 batch_size=128, 
                                                 shuffle=False,
                                                 drop_last=False)        
        trainer.validate(model, dataloaders=val_loader)
    else: # start training routine
        
        val_loader = torch.utils.data.DataLoader(dataset=DH.SequenceDataValidation(user_train, 
                                                                                   user_valid, 
                                                                                   usernum, 
                                                                                   itemnum, 
                                                                                   args['maxlen'], 
                                                                                   args['ndcg_samples']),
                                                 batch_size=args['batch_size'], 
                                                 shuffle=True,
                                                 drop_last=True)
    
        test_loader = torch.utils.data.DataLoader(dataset=DH.SequenceDataTest(user_train, 
                                                                              user_valid, 
                                                                              user_test, 
                                                                              usernum, 
                                                                              itemnum, 
                                                                              args['maxlen'],
                                                                              args['ndcg_samples']),
                                                  batch_size=args['batch_size'], shuffle=False,
                                                  drop_last=True)

        train_loader = torch.utils.data.DataLoader(dataset=DH.SequenceData(user_train, usernum, itemnum),
                                                   batch_size=args['batch_size'],
                                                   shuffle=True,
                                                   collate_fn=DH.tokenize_batch)        
        
        
        if args['opt'] == 'FusedAdam':
            try:
                import apex
            except ModuleNotFoundError:
                print("\n >>>No apex installed - switching to simple Adam<<<\n")
                args['opt'] = 'Adam'
        model = SASRecEncoder(itemnum, **args)
        
        if args['xavier_init']:
        # weight initialization
            print("\nRunning weights initialization with xavier normal...\n")
            for name, param in model.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                    print(f"{name:<40} sucess")
                except:
                    print(f"{name:<40} failure")

        trainer.fit(model, train_loader, val_loader)

        torch.save(model.state_dict(), f"sasrec_{trainer.logger.version}.pt")

        # metrics on test dataset
        trainer.test(model, test_loader)
