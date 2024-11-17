"""
Train a diffusion model on images.
"""
import random
import logging
import sys
import datetime
import argparse
import json, torch, os
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer,selfTokenizer
from mydatasets import get_dataloader,ChEBIdataset
import warnings
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")

def main_worker(rank,world_size,seed):

    #将命令行输入的参数解析成一个对象 args，以便后续在代码中使用这些参数的值。
    args = create_argparser(seed).parse_args()
    #

    # set_seed(args.seed)
    # logging.info("seed:"args.seed)

    #添加，加入取消wandb上传记录的代码
    #os.environ["WANDB_MODE"] = "dryrun"
    #添加

    # 这段代码的作用是在 rank == 0 的进程中初始化 wandb 实验，并打印出实验的参数配置信息。
    # if rank == 0:
    #     wandb.init(
    #         project = "DiffusionLMRegexAug",
    #         config = args.__dict__,
    #     )
    #     print(wandb.config)
    # #

    #设置了当前进程的编号和总进程数，为后续分布式训练做准备
    dist_util.setup_dist(rank,world_size)
    #
    print("creating model and diffusion...")

    #正则表达式分词器对象，用于对输入文本进行分词处理，并设置了最大长度为 256。
    smtokenizer = regexTokenizer(max_len=256)
    sftokenizer = selfTokenizer(max_len=256)
    #
    if not args.is_self:
        vocab_size=len(smtokenizer)
    else:
        vocab_size=len(sftokenizer)
    model = TransformerNetModel2(
        in_channels=args.model_in_channels,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=args.model_model_channels,
        dropout=args.model_dropout,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=vocab_size,
        experiment_mode='lm',
        logits_mode=1,
        hidden_size = args.model_hidden_size,
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers = args.model_num_hidden_layers,
    )

    if rank==0:
        #这行代码计算模型参数的总数。通过遍历模型的参数并调用 numel() 方法获取参数的元素个数，然后将所有参数的元素个数相加，得到模型参数的总数
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        #
        print('Model total prameter number is :', pytorch_total_params)
        #这行代码输出分词器 smtokenizer 的词汇表长度。
        print('Smiles tokenizer vocab length:',len(smtokenizer))
        #
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(2000)],
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=(
             gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
            )
        ),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
    )
    if args.continue_ == True:
        print(args.model_path)
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        logging.info("continue....")
        logging.info(args.model_path)
    else:
        logging.info("not continue")
    #用于定义扩散过程中时间步的采样策略。在这里，使用了均匀采样策略。
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    #
    #表示正在加载数据
    print('load data', '*'*50)
    #
    if not args.is_self:
        dir='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/SMILES'
    else:
        dir='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/SELFIES'
    #创建了一个名为 train_dataset 的数据集对象，用于加载训练数据。在这里，指定了数据集的目录、分词器、数据集划分方式、是否替换描述、损坏概率和是否掩盖描述等参数。
    train_dataset = ChEBIdataset(
        is_self = args.is_self,
        #修改，修改dir为绝对路径
        #dir='../../datasets/SMILES/',
        dir=dir,
        self_tokenizer=sftokenizer,
        smi_tokenizer=smtokenizer,
        split='train_val_256',
        replace_desc=False,
        corrupt_prob=0.,
        mask_desc=False
        # pre = pre
    )
    #

    print('In total',len(train_dataset),'for training....')
    #创建了一个名为 dataloader 的数据加载器对象，用于批量加载训练数据。在这里，传入了数据集对象、批量大小、当前进程的编号和总进程数等参数
    dataloader = get_dataloader(train_dataset,args.batch_size,rank,world_size)
    #

    #data_valid = None: 定义了一个用于验证的数据集，这里设置为 None，表示不进行验证
    data_valid = None
    #
    #创建了一3个名为 TrainLoop 的训练循环对象，并传入了训练所需的各种参数，包括模型、扩散过程、数据加载器、批量大小、学习率等。
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        continue_=args.continue_,
        model_path_num=args.model_path_num,
        seed = seed,
        is_for_seed = args.is_for_seed,
    ).run_loop()

    #释放资源
    dist.destroy_process_group()


def create_argparser(in_seed):
    defaults = dict()
    text_defaults = dict(
        attention_resolutions='16,8', 
        #cuda内存不够调小了batch
        #batch_size=64, 
        batch_size=64, 
        cache_mode='no', 
        checkpoint_path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/checkpoints_self', 
        #checkpoint_path='../../checkpoints', 
        class_cond=False, 
        commonGen_train='diffusion_lm/common-gen/commongen_data', 
        config='ll', 
        config_name='bert-base-uncased', 
        data_dir='', 
        dataset_config_name='wikitext-2-raw-v1', 
        dataset_name='wikitext', 
        diffusion_steps=200000, 
        dropout=0.1, 
        e2e_train='', 
        ema_rate='0.9999', 
        emb_scale_factor=1.0, 
        eval_interval=2000, 
        experiment='random', 
        experiment_mode='lm', 
        fp16_scale_growth=0.001, 
        gradient_clipping=2.4, 
        image_size=8, 
        in_channel=16, 
        learn_sigma=False, 
        log_interval=20, 
        logits_mode=1, 
        lr=0.0001, 
        #控制运行多少步后停止运行
        #self.step + self.resume_step < self.lr_anneal_steps//self.world_size
        lr_anneal_steps=200000, 
        microbatch=-1, 
        modality='e2e-tgt', 
        model_arch='transformer', 
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None', 
        noise_level=0.0, 
        noise_schedule='sqrt', 
        num_channels=128, 
        num_heads=4, 
        num_heads_upsample=-1, 
        num_res_blocks=2, 
        out_channel=16, 
        padding_mode='pad', 
        predict_xstart=True, 
        preprocessing_num_workers=1, 
        rescale_learned_sigmas=True, 
        rescale_timesteps=True, 
        resume_checkpoint='', 
        roc_train='diffusion_lm/ROCstory', 
        save_interval=100000, 
        schedule_sampler='uniform', 
        seed=in_seed, 
        sigma_small=False, 
        timestep_respacing='', 
        training_mode='e2e', 
        use_bert_tokenizer='no', 
        use_checkpoint=False, 
        use_fp16=False, 
        use_kl=False, 
        use_scale_shift_norm=True, 
        weight_decay=0.0,
        model_in_channels = 32,
        model_model_channels = 128,
        model_dropout = 0.1,
        model_hidden_size = 1024,
        model_num_attention_heads = 16,
        model_num_hidden_layers = 12,
        model_path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/checkpoints_self/PLAIN_ema_0.9999_000650.pt',
        model_path_num=0,
        continue_=False,
        is_self =True,
        is_for_seed = True,
    )
    text_defaults['model_path_num'] = int(text_defaults['model_path'].split('_')[-1].split('.')[0])
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


#add
# 定义输出重定向函数
def redirect_output_to_file(filename):
    sys.stdout = open(filename, 'a')

# 恢复标准输出
def restore_output():
    sys.stdout.close()
    sys.stdout = sys.__stdout__

# if __name__ == "__main__":
#     print("ss")
#     os.environ["CUDA_VISIBLE_DEVICES"]="1"
#     import os

#     #os.environ['CUDA_DEVICES_ORDER']='PCI_BUS_ID'
#     world_size=1
#     mp.spawn(main_worker,args=(world_size,),nprocs=world_size,join=True)

if __name__ == "__main__":
    train_for_seed = True
    if train_for_seed:
        logging.basicConfig(filename='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/improved-diffusion/log/seeds.log', level=logging.INFO)
        logging.info("开始寻找seed...")
        # 设置要训练的次数
        num_trainings = 20
        seed = 20066206
        for i in range(num_trainings):
            seed = seed + random.randint(0,10000)
            set_seed(seed)
            logging.info(f"当前训练次数：{i + 1}，种子：{seed}")
            # 运行主工作函数
            world_size = 1
            start_time = datetime.datetime.now()
            main_worker(0, world_size,seed)
            end_time = datetime.datetime.now()
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = end_time - start_time
            days = duration.days
            seconds = duration.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            logging.info(f"程序开始时间：{start_time_str}")
            logging.info(f"程序结束时间：{end_time_str}")
            logging.info(f"程序运行时间：{days} 天 {hours} 小时 {minutes} 分钟 {seconds} 秒")
            print(f"程序运行时间：{days} 天 {hours} 小时 {minutes} 分钟 {seconds} 秒")
            
    else:
        logging.basicConfig(filename='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/improved-diffusion/log/training.log', level=logging.INFO)
        logging.info("开始训练模型1...")
        print("ss")
        #os.environ["CUDA_VISIBLE_DEVICES"]="0"
        set_seed(20044070)
        logging.info("seed:20044070")
        # 设置单个进程
        world_size = 1
        start_time = datetime.datetime.now()
        main_worker(0, world_size)
        end_time = datetime.datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        # 计算程序运行时间
        duration = end_time - start_time
        days = duration.days
        seconds = duration.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        logging.info(f"程序开始时间：{start_time_str}")
        logging.info(f"程序结束时间：{end_time_str}")
        logging.info(f"程序运行时间：{days} 天 {hours} 小时 {minutes} 分钟 {seconds} 秒")
        print(f"程序运行时间：{days} 天 {hours} 小时 {minutes} 分钟 {seconds} 秒")