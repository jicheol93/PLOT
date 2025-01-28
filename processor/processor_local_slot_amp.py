import logging
import time
import datetime
import torch
from utils.meter import AverageMeter
from utils.metrics_local_slot import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

##from PCGrad.pcgrad import PCGrad

import pdb

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, logger):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logger
    logger.propagate = False
 
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "metric_loss": AverageMeter(),
        "metric_local_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "id_local_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "pmlm_loss": AverageMeter(),
        "recon_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "img_local_acc": AverageMeter(),
        "txt_local_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    scaler = torch.cuda.amp.GradScaler()

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_eval_time = time.time()
        start_time = time.time()
        ##JC
        ##top1 = evaluator.eval(model.eval(), tb_writer, epoch=epoch)
        ##
        total_eval_time = time.time() - start_eval_time
        total_eval_time_str = str(datetime.timedelta(seconds=int(total_eval_time)))
        print('Retrieval time {}'.format(total_eval_time_str))
        for meter in meters.values():
            meter.reset()
        model.train()
        print(model.base_model.transformer.resblocks[-1].attn.out_proj.weight.requires_grad)

        for n_iter, batch in enumerate(train_loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                batch = {k: v.to(device) for k, v in batch.items()}

                ret = model(batch,epoch)
                total_loss = sum([v for k, v in ret.items() if "loss" in k])

                batch_size = batch['images'].shape[0]
                meters['loss'].update(total_loss.item(), batch_size)
                meters['metric_loss'].update(ret.get('metric_loss', 0), batch_size)
                meters['metric_local_loss'].update(ret.get('metric_local_loss', 0), batch_size)
                meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
                meters['id_local_loss'].update(ret.get('id_local_loss', 0), batch_size)
                meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
                meters['pmlm_loss'].update(ret.get('pmlm_loss', 0), batch_size)
                meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
                meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
                meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
                meters['img_local_acc'].update(ret.get('img_local_acc', 0), batch_size)
                meters['txt_local_acc'].update(ret.get('txt_local_acc', 0), batch_size)
                meters['recon_loss'].update(ret.get('recon_loss', 0), batch_size)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval(), tb_writer, epoch=epoch)
                else:
                    top1 = evaluator.eval(model.eval(), tb_writer, epoch=epoch)

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, args):

    logger = logging.getLogger("PLOT.test")
    logger.info("Enter inferencing")

    tb_writer = SummaryWriter(log_dir=args.output_dir, filename_suffix='test')


    start_eval_time = time.time()
    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    top1 = evaluator.eval(model.eval(), tb_writer, epoch=999)
    total_eval_time = time.time() - start_eval_time
    total_eval_time_str = str(datetime.timedelta(seconds=int(total_eval_time)))
    print('Retrieval time {}'.format(total_eval_time_str))
