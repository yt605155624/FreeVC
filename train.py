import logging

logger = logging.getLogger('matplotlib')
logger.setLevel(logging.INFO)
logger = logging.getLogger('torch.nn.parallel.distributed')
logger.setLevel(logging.WARNING)
logger = logging.getLogger('torch.distributed.distributed_c10d')
logger.setLevel(logging.WARNING)

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import commons
import utils
from data_utils import TextAudioSpeakerLoader
from data_utils import TextAudioSpeakerCollate
from data_utils import DistributedBucketSampler
from models import SynthesizerTrn

from models import MultiPeriodDiscriminator
from losses import generator_loss
from losses import discriminator_loss

from losses import feature_loss

from losses import kl_loss

from mel_processing import mel_spectrogram_torch
from mel_processing import spec_to_mel_torch

# 设置为 False 后会使得显存增高导致 OOM, bs = 256 相对有点大
"""
如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
"""
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
global_step = 0
global_total_epoch = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, ))
    # run(0,1,hps)


def run(rank, n_gpus, hps):
    global global_step
    global global_total_epoch
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    # torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate(hps)
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=True,
            batch_size=hps.train.eval_batch_size,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn)

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])
    try:
        _, _, _, epoch_str, iter_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
            optim_g)
        _, _, _, epoch_str, iter_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
            optim_d)
        global_step = iter_str
    except Exception:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    global_total_epoch = hps.train.epochs

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank=rank,
                epoch=epoch,
                hps=hps,
                nets=[net_g, net_d],
                optims=[optim_g, optim_d],
                schedulers=[scheduler_g, scheduler_d],
                scaler=scaler,
                loaders=[train_loader, eval_loader],
                logger=logger,
                writers=[writer, writer_eval])
        else:
            train_and_evaluate(
                rank=rank,
                epoch=epoch,
                hps=hps,
                nets=[net_g, net_d],
                optims=[optim_g, optim_d],
                schedulers=[scheduler_g, scheduler_d],
                scaler=scaler,
                loaders=[train_loader, None],
                logger=None,
                writers=None)
        scheduler_g.step()
        scheduler_d.step()
    # 释放进程组资源
    dist.destroy_process_group()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler,
                       loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        # 本条数据有问题, 跳过
        if None in items:
            logger.info("skip one batch of data because of data error!")
            global_step += 1
            continue
        if hps.model.use_spk:
            c, spec, y, spk = items
            g = spk.cuda(rank, non_blocking=True)
        else:
            c, spec, y = items
            g = None

        spec, y = spec.cuda(
            rank, non_blocking=True), y.cuda(
                rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        mel = spec_to_mel_torch(spec, hps.data.filter_length,
                                hps.data.n_mel_channels, hps.data.sampling_rate,
                                hps.data.mel_fmin, hps.data.mel_fmax)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q,
                                       logs_q) = net_g(
                                           c, spec, g=g, mel=mel)

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin,
                hps.data.mel_fmax)
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                                       hps.train.segment_size)  # slice 

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                # 这个 l1 loss 是不是错了？ 应该用 spec 计算？
                # -> 没错， paddlespeech 里面是 speech 转成 mel 再计算
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                  z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                # current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                #                              time.localtime(time.time()))
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch, 100. * batch_idx / len(train_loader)) +
                            " / Total Epoch: " + str(global_total_epoch))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g
                }
                scalar_dict.update({
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl
                })

                scalar_dict.update({
                    "loss/g/{}".format(i): v
                    for i, v in enumerate(losses_gen)
                })
                scalar_dict.update({
                    "loss/d_r/{}".format(i): v
                    for i, v in enumerate(losses_disc_r)
                })
                scalar_dict.update({
                    "loss/d_g/{}".format(i): v
                    for i, v in enumerate(losses_disc_g)
                })
                image_dict = {
                    "slice/mel_org":
                    utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen":
                    utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()),
                    "all/mel":
                    utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)
            if global_step % hps.train.eval_interval == 0:
                evaluate(
                    hps=hps,
                    net_g=net_g,
                    net_d=net_d,
                    eval_loader=eval_loader,
                    writer_eval=writer_eval,
                    logger=logger)
                utils.save_checkpoint(
                    model=net_g,
                    optimizer=optim_g,
                    learning_rate=hps.train.learning_rate,
                    iteration=global_step,
                    epoch=epoch,
                    checkpoint_path=os.path.join(
                        hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(
                    model=net_d,
                    optimizer=optim_d,
                    learning_rate=hps.train.learning_rate,
                    iteration=global_step,
                    epoch=epoch,
                    checkpoint_path=os.path.join(
                        hps.model_dir, "D_{}.pth".format(global_step)))

        global_step += 1
        # if rank == 0:
        #     print("global_step:", global_step, "current_time: ",
        #           time.strftime('%Y-%m-%d %H:%M:%S',
        #                         time.localtime(time.time())))

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, net_g, net_d, eval_loader, writer_eval, logger):
    net_g.eval()
    net_d.eval()
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            if hps.model.use_spk:
                c, spec, y, spk = items
                g = spk.cuda(0)
            else:
                c, spec, y = items
                g = None
            break

        c, spec, y = c.cuda(0), spec.cuda(0), y.cuda(0)
        mel = spec_to_mel_torch(spec, hps.data.filter_length,
                                hps.data.n_mel_channels, hps.data.sampling_rate,
                                hps.data.mel_fmin, hps.data.mel_fmax)
        y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
            c, spec, g=g, mel=mel)
        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size //
                                       hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            hps.data.mel_fmin, hps.data.mel_fmax)
        y = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                                   hps.train.segment_size)  # slice
        # Discriminator loss
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
        # Gen loss
        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                  z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info("Eval: " + str([x.item() for x in losses]))
        eval_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc_all,
        }
        eval_dict.update({
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl
        })

        eval_dict.update(
            {"loss/g/{}".format(i): v
             for i, v in enumerate(losses_gen)})
        eval_dict.update(
            {"loss/d_r/{}".format(i): v
             for i, v in enumerate(losses_disc_r)})
        eval_dict.update(
            {"loss/d_g/{}".format(i): v
             for i, v in enumerate(losses_disc_g)})
        # 注意这里用 infer 上面求 loss 用 forward
        # 上面求 loss 的 gt mel 是 y_mel, 是经过 slice 的
        # 这里的 y_hat 应该比上面的 y_hat 长
        # 所以这里是和 mel 比较, 上面是和 y_mel 求 l1 loss
        if hps.model.use_spk:
            g = spk[:1].cuda(0)
        # 取 batch 里面的第一条
        c, spec, y = c[:1].cuda(0), spec[:1].cuda(0), y[:1].cuda(0)
        # 这里不 slice 所以不需要考虑原始长度的 segment_size 的关系
        y_hat = net_g.module.infer(c, g=g, mel=mel)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(), hps.data.filter_length,
            hps.data.n_mel_channels, hps.data.sampling_rate,
            hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin,
            hps.data.mel_fmax)

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0], "gt/audio": y[0]}
    utils.summarize(
        writer=writer_eval,
        scalars=eval_dict,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate)
    net_g.train()
    net_d.train()


if __name__ == "__main__":
    main()
