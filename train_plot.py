import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor_local_slot_amp import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model_local_slot_detachRecon
from utils.metrics_local_slot import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

import pdb

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    ##set_seed(1+get_rank())
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cur_time_tilD = cur_time.split('_')[0]
    ##args.output_dir = op.join(args.output_dir, args.dataset_name, f'{name}_{cur_time}')
    ##args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time_tilD}_{name}')
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{name}')
    logger = setup_logger('PLOT', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    if args.plugin:
        model = build_model_local_slot_plugin(args, num_classes)
    elif args.inner:
        model = build_model_local_slot_inner(args, num_classes)
    elif args.localonly:
        model = build_model_local_slot_only(args, num_classes)
    elif args.detach_recon:
        model = build_model_local_slot_detachRecon(args, num_classes)
    elif args.tipcb:
        model = build_model_local_slot_tipcb(args, num_classes)
    else:
        model = build_model_local_slot(args, num_classes)

    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, args)

    start_epoch = 1

    if args.plugin:
        f = "/home/jicheol/project_nlps/IRRA/logs/CUHK-PEDES/20230718_212104_iira/best.pth"
        ##f = "/home/jicheol/project_nlps/IRRA/logs/CUHK-PEDES/20230921_131218_IRRA_rep/best.pth"
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        ##model.irra.load_state_dict(checkpoint.pop("model"))
        checkpoint_dict = checkpoint.pop("model")
        
        pre_dict = {k.replace('base_model.',''): v for k, v in checkpoint_dict.items() if 'base_model' in k}
        model.base_model.load_state_dict(pre_dict)
        for param in model.base_model.parameters():
            param.requires_grad = False

    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer, logger)
