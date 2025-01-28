import torch

from .lr_scheduler import LRSchedulerWithWarmup

import pdb

def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        
        ##manual_lr_factor = 20.0

        """
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        """
        if "slot_attention" in key:
            lr = args.lr*args.lr_factor
            ##lr = args.lr*0.1
        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
            ##lr =  args.lr*1.0 # default 5.0
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor
            ##lr = args.lr * 1.0
        if "recon_decoder" in key:
            lr = args.lr * args.lr_factor
        if "slots" in key:
            lr = args.lr * args.lr_factor
        if "part" in key:
            lr = args.lr * args.lr_factor
        if "local" in key or "slot" in key or "decoder" in key:
            lr = args.lr * args.lr_factor
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
            ##lr = args.lr*args.lr_factor

        """
        if "base_model" in key:
            lr = args.lr*0.1
        if "encode_text" in key:
            lr = args.lr*0.0
        """

       
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
