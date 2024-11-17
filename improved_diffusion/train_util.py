import copy
import functools
import os
import logging
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import random
from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import wandb
import pickle
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    # 初始化训练循环对象，设置了训练所需的各种参数，包括模型、扩散过程、数据加载器、批量大小、学习率等。
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=-1,
        continue_=False,
        model_path_num=0,
        seed=1420405,
        is_continue=False,
        opt_path='',
        load_num=0,
        num=0,
        is_for_seed=False,
        is_join=False,
        is_two_way=False,
    ):
        print("IN AUG trainutil")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print("initialing Trainer for",rank,'/',world_size)
        self.num = num
        self.seed = seed
        self.continue_ = continue_
        self.model_path_num = model_path_num
        self.rank = rank
        self.world_size = world_size
        self.diffusion = diffusion
        self.data = data
        self.is_join = is_join
        self.is_continue = is_continue
        self.is_two_way = is_two_way
        self.opt_path = opt_path
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.load_num = load_num
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr*world_size
        self.is_for_seed = is_for_seed
        print("ori lr:",lr,"new lr:",self.lr)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion) #用于生成均匀分布的采样权重
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.gradient_clipping = gradient_clipping
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        print('checkpoint_path:{}'.format(checkpoint_path))
        self.checkpoint_path = checkpoint_path # DEBUG **
        
        self.model = model.to(rank)
       
        # self._load_and_sync_parameters()
        # if self.use_fp16:
        #     self._setup_fp16()

        
        

        if th.cuda.is_available(): # DEBUG **
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.rank],
                # device_ids=[dist_util.dev()],
                # output_device=dist_util.dev(),
                # broadcast_buffers=False,
                # bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            assert False
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            # self.use_ddp = False
            # self.ddp_model = self.model
        self.model_params = list(self.ddp_model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            # self._load_optimizer_state()
            # # Model was resumed, either due to a restart or a checkpoint
            # # being specified at the command line.
            # self.ema_params = [
            #     self._load_ema_parameters(rate) for rate in self.ema_rate
            # ]
            pass
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
        if self.is_continue:
            self.opt.load_state_dict(dist_util.load_state_dict(self.opt_path))
            path = self.opt_path
            name = path.split('/')[-1]  # 获取文件名
            num = int(name.split('_')[0])  # 获取模型索引
            if num!=0:
                print("num",num)
                for i in range(0,num):
                    next(self.data)
            self.step = num
            self.num = num

            


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                # logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = self.opt_path
        if bf.exists(opt_checkpoint):
            logging.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(th.load(self.opt_path))
            logging.info("load opt")

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    # 执行训练循环，不断地进行训练步骤，直到达到指定的训练步数或条件。
    def run_loop(self):
        print('START LOOP FLAG')
        if self.is_for_seed:
            while (
                # 判断是否进行学习率的退火
                not self.lr_anneal_steps
                or self.step + self.resume_step < 2000
            ):
                batch = next(self.data)
                cond = None
                if self.step != self.num or self.num==0:
                    self.run_step(batch, cond)
                if self.step % self.log_interval == 0:
                    pass
                if self.eval_data is not None and self.step % self.eval_interval == 0:
                    print('eval on validation set')
                    pass
                if self.step % self.save_interval == 0 and self.step!=0:
                    if self.step != self.num or self.num==0:
                        logging.info("saving")
                        self.save()
                        print("step",self.step)
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                self.step += 1
        else:
            while (
                # 判断是否进行学习率的退火
                not self.lr_anneal_steps
                #
                or self.step + self.resume_step < self.lr_anneal_steps//self.world_size
                # or self.step + self.resume_step < 2000
            ):
                # 从数据加载器中获取一个批次的数据,数据中包含编码后的smiles，文字描述，mask和编码后的smiles
                # torch.Size([batch, 256])
                # torch.Size([batch, 216, 768])
                # torch.Size([batch, 216])
                # torch.Size([batch, 256])
                batch = next(self.data)
                cond = None
                # if self.step<3:
                #     print("RANK:",self.rank,"STEP:",self.step,"BATCH:",batch)

                # 执行一个训练步骤
                if self.step != self.num or self.num==0:
                    self.run_step(batch, cond)

                # 如果当前步数可以被 log_interval 整除，则执行下面的代码块
                if self.step % self.log_interval == 0:
                    # dist.barrier()
                    pass
                    #logger.dumpkvs()

                # 如果评估数据集不为空且当前步数可以被 eval_interval 整除，则执行下面的代码块，用于在验证集上进行评估
                if self.eval_data is not None and self.step % self.eval_interval == 0:
                    # batch_eval, cond_eval = next(self.eval_data)
                    # self.forward_only(batch, cond)
                    print('eval on validation set')
                    pass# logger.dumpkvs()

                # 如果当前步数可以被 save_interval 整除且不等于0，则执行下面的代码块，用于保存模型参数
                if self.step % self.save_interval == 0 and self.step!=0:
                    if self.step != self.num or self.num==0:
                        logging.info("saving")
                        self.save()
                        print("step",self.step)
                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                self.step += 1
                # Save the last checkpoint if it wasn't already saved.
            if (self.step - 1) % self.save_interval != 0:
                logging.info("saving")
                self.save()

    # 执行训练的一步，包括前向传播、反向传播和优化器更新。
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            #混合精度（FP16）优化
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_only(self, batch, cond):
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )
            
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )

    # 执行一步中的前向传播和反向传播过程，计算损失并进行梯度计算和参数更新。
    def forward_backward(self, batch, cond):
        # zero_grad(self.model_params)
        self.opt.zero_grad()
        for i in range(0, batch[0].shape[0], self.microbatch):
            # micro = batch[i : i + self.microbatch].to(self.rank)
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], self.rank)
            try:
                micro = (batch[0].to(self.rank),batch[1].to(self.rank),batch[2].to(self.rank),batch[3].to(self.rank),batch[4].to(self.rank))
                print("micro : 4")
            except:
                micro = (batch[0].to(self.rank),batch[1].to(self.rank),batch[2].to(self.rank),batch[3].to(self.rank))
                print("micro : 3")
            last_batch = True
            if self.step == self.num or self.num !=0:
                for i in range(0,self.num+1):
                    t, weights = self.schedule_sampler.sample(micro[0].shape[0], self.rank)
            t, weights = self.schedule_sampler.sample(micro[0].shape[0], self.rank)
            t.shape

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=None,
            )
            if not self.is_join:
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                # print('----DEBUG-----',self.step,self.log_interval)
                if self.step % self.log_interval == 0 and self.rank==0:
                    print("rank0: ",self.step,loss.item())
                    # wandb.log({'loss':loss.item()})
                    msg = f'step:{self.step},loss:{loss.item()}'
                    logging.info({msg})
            else:
                if last_batch or not self.use_ddp:
                    losses,loss_add = compute_losses()
                    
                else:
                    with self.ddp_model.no_sync():
                        losses,loss_add = compute_losses()
                loss_add.detach()
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                loss_add = (loss_add * weights).mean()
                # print('----DEBUG-----',self.step,self.log_interval)
                if self.step % self.log_interval == 0 and self.rank==0:
                    print("rank0: ",self.step,loss.item())
                    # wandb.log({'loss':loss.item()})
                    msg = f'step:{self.step},loss:{loss.item()},loss_add:{loss_add},loss_ori:{loss.item()-loss_add}'
                    logging.info({msg})
            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )
            if self.use_fp16:
                # loss_scale = 2 ** self.lg_loss_scale
                # (loss * loss_scale).backward()
                pass
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm=self.gradient_clipping #3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(), #amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        # self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        # logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        # logger.logkv("step", self.step + self.resume_step)
        # msg = "step: {}".format(self.step + self.resume_step)
        # logging.info(msg)
        # msg = "step: {}".format(self.step + self.resume_step)
        # logging.info(msg)
        # msg = "samples: {}".format((self.step + self.resume_step + 1) * self.global_batch)
        # logging.info(msg)
        # logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            # logger.logkv("lg_loss_scale", self.lg_loss_scale)
            msg = "lg_loss_scale: {}".format(self.lg_loss_scale)
            # logging.info(msg)
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if self.continue_:
                start = self.model_path_num
            else:
                start = 0
            if dist.get_rank() == 0:
                # logger.log(f"saving model {rate}...")
                print(f"saving model {rate}...")
                for param_group in self.opt.param_groups:
                    lr = param_group["lr"]
                if not rate:
                    filename = f"{lr}_{self.seed}_PLAIN_model_{((self.step+self.resume_step)*self.world_size):06d}.pt"
                    opt_path = f"{self.step}_{self.seed}_opt_ema_{((self.step+self.resume_step)*self.world_size):06d}.pt"
                else:
                    filename = f"{lr}_{self.seed}_PLAIN_ema_{rate}_{((self.step+self.resume_step)*self.world_size):06d}.pt"
                    opt_path = f"{self.step}_{self.seed}_opt_ema_{rate}_{((self.step+self.resume_step)*self.world_size):06d}.pt"
                logging.info(filename)
                if self.step == self.lr_anneal_steps:
                    th.save(self.opt.state_dict(), bf.join(self.checkpoint_path, opt_path))
                # print('writing to', bf.join(get_blob_logdir(), filename))
                # print('writing to', bf.join(self.checkpoint_path, filename))
                # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                #     th.save(state_dict, f)
                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f: # DEBUG **
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0: # DEBUG **
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    return
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

