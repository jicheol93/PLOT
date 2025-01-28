from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor_local_slot_amp import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model_local_slot_detachRecon
from utils.metrics_local_slot import Evaluator
import argparse
from utils.iotools import load_train_configs
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PLOT Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/plot/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('PLOT', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model_local_slot_detachRecon(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader, args)
