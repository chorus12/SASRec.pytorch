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
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.nn import MultiheadAttention, LayerNorm, Dropout, Conv1d, Embedding, BCEWithLogitsLoss
from SASRecModel import PointWiseFF, SASRecEncoderLayer, PositinalEncoder, SASRecEncoder


# setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--optimizer', default='Adam', type=str)
parser.add_argument('--num_epochs', default=300, type=int)

parser.add_argument('--hidden_units', default=50,
                    type=int)  # synonym for d_model
parser.add_argument('--d_model', default=50, type=int)  # same as hidden_units

parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=bool)
parser.add_argument('--state_dict_path', default=None, type=str)

# args = parser.parse_args(['--dataset=ml-1m', '--train_dir=default',
#                           '--maxlen=200', '--dropout_rate=0.2', '--device=cuda'])

args = parser.parse_args()
args = vars(args)


if __name__ == '__main__':
    # read dataset
    dataset = DH.data_partition(args['dataset'])

    print('Runtime parameters\n',*[(k, v) for (k, v) in args.items()], sep="\n")

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # batches got sliced by users, i.e. batch accumulate BATCH_SIZE user sequences of items selected/bought
    BATCH_SIZE = args['batch_size']
    num_batch = len(user_train) // BATCH_SIZE  # number of batches

    user_train_lens = list(map(len, [v for k, v in user_train.items()]))
    print(
        f'average sequence length: {sum(user_train_lens)/len(user_train):.1f}')

    print(f"\nBatch size is - {args['batch_size']}\n")

    val_loader = torch.utils.data.DataLoader(dataset=DH.SequenceDataValidation(user_train, user_valid, usernum, itemnum, args['maxlen']),
                                             batch_size=args['batch_size'], shuffle=True,
                                             drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=DH.SequenceDataTest(user_train, user_valid, user_test, usernum, itemnum, args['maxlen']),
                                              batch_size=args['batch_size'], shuffle=True,
                                              drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=DH.SequenceData(user_train, usernum, itemnum),
                                               batch_size=args['batch_size'],
                                               shuffle=True,
                                               collate_fn=DH.tokenize_batch)

    model = SASRecEncoder(itemnum, opt=args['optimizer'], **args)

    # weight initialization
    print("\nRunning weights initialization with xavier normal...\n")
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
            print(f"{name:<40} sucess")
        except:
            print(f"{name:<40} failure")

    # save checkpoints
    checkpoint_callback = ModelCheckpoint(monitor="HR@10/val")

    # run tensorboard before the script launch
    # tensorboard --logdir ./lightning_logs/ --host 0.0.0.0

    # run the script with the command
    # PL_TORCH_DISTRIBUTED_BACKEND=nccl python SASRemMain.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda

    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = pl.Trainer(gpus=-1,
                         auto_select_gpus=False,
                         max_epochs=args['num_epochs'],
                         reload_dataloaders_every_n_epochs=1,
                         val_check_interval=1.0,
                         callbacks=[checkpoint_callback],
                         # log 4 times per epoch
                         log_every_n_steps=int(
                             len(train_loader.dataset) / args['batch_size'] / 3),
                         # limit_val_batches=0, How much of validation dataset to check. Useful when debugging or testing something that happens at the end of an epoch.
                         num_sanity_val_steps=10,
                         precision=16)

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), "bazman_sasrec_refactored.pt")

    # metrics on test dataset
    trainer.test(model, test_loader)
