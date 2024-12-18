import argparse
import os, json
import logging
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from mydatasets import get_dataloader,ChEBIdataset

def main():
    logging.basicConfig(filename='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/improved-diffusion/log/post_sample_log.log', level=logging.INFO)
    logging.info("开始post_sample")
    set_seed(121)
    args = create_argparser().parse_args()

    
    args.sigma_small = True


    if args.experiment == 'random1': args.experiment = 'random'
    logging.info("creating model and diffusion...")
    from mytokenizers import regexTokenizer
    tokenizer = regexTokenizer(max_len=args.ml)
    model = TransformerNetModel2(
        in_channels=32,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        # config_name='../../bert-base-uncased',
        config_name='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/improved-diffusion/bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(tokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size = 1024,
        num_attention_heads=16,
        num_hidden_layers = 12,
    )
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0,400,20)],
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

    print(args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'the parameter count is {pytorch_total_params}')

    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval() # DEBUG

    logging.info("sampling...")

    # with open('../../tempbadmols.txt') as f:
    with open('/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/tempbadmols.txt') as f:
        content = [l.strip().split() for l in f.readlines()]
        mols  = [l[1] for l in content]
        orders = [l[0] for l in content]

    args.num_samples = len(mols)
    print(args.num_samples)

    noise = model.word_embedding(tokenizer(mols).to(dist_util.dev()))
    print(noise.shape)
    allsample = []
    num_done = 0
    while num_done < args.num_samples:
        idend = min(num_done+args.batch_size,args.num_samples)
        print('acquiring  {} : {}'.format(num_done,idend))
        startnoise = noise[num_done:idend]
        desc_state = th.zeros(idend-num_done,200,768)
        desc_mask = th.ones(idend-num_done,200)
        
        model_kwargs = {}
        print('use_ddim:{}',args.use_ddim)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_shape = (idend-num_done, 200, model.in_channels)
        print(sample_shape)
        sample = sample_fn(
            model,
            sample_shape,
            noise = startnoise,
            clip_denoised=args.clip_denoised,
            denoised_fn = None,
            model_kwargs=model_kwargs,
            top_p =args.top_p,
            progress = True,
            desc = (desc_state,desc_mask)
        )
        allsample.append(sample)
        num_done = idend
    sample = th.concat(allsample,dim=0)
    print('decoding for e2e', )
    print(sample.shape)
    x_t = th.tensor(sample).cuda()
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices
    sample = sample.squeeze(-1)
    print(sample)
    tokenizer = regexTokenizer(max_len=args.ml)
    c = tokenizer.decode(sample)
    with open(args.outputdir,'w') as f:
        for i,x in enumerate(c):
            if i==0:
                print(x)
            f.write(orders[i]+'\t'+x.replace('[PAD]','')+'\n')


    # with open('../../textguidtry_256_final.txt') as f:
    with open('/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/textguidtry_256_final.txt') as f:
        content = [k.strip().split('   ||   ') for k in f.readlines()]
    with open(args.outputdir) as f:
        tochange = [k.strip().split() for k in f.readlines()]
    from rdkit import Chem
    f = Chem.MolFromSmiles
    changecnt = 0
    for (num,smiles) in tochange:
        mol = f(smiles.replace('[SOS]','').replace('[EOS]',''))
        if mol is None: 
            content[int(num)][0]=smiles
            changecnt+=1

    with open('/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/tempoutput.txt','w') as f:
        for c in content:
            f.write(c[0]+'   ||   '+c[1]+'\n')
    print('Repaired {}'.format(changecnt))

    with open('/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/tempoutput.txt') as f:
    # with open('../../tempoutput.txt') as f:
        x = f.readlines()
    output = [i.strip().split('   ||   ')[0].replace('[SOS]','').replace('[EOS]','') for i in x]
    # with open('../../datasets/SMILES/test.txt') as f:
    with open('/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/SMILES/test.txt') as f:
        x = f.readlines()[1:]
    ground = [i.strip().split('\t')[1] for i in x if len(i)>3]
    description = [i.strip().split('\t')[2] for i in x if len(i)>3]
    assert(len(output)==len(ground)==len(description))
    # with open('../../MODELOUTPUT.txt','w') as f:
    logging.info("begin writing")
    with open('/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/MODELOUTPUT.txt','w') as f:
        f.write('description\tground truth\toutput\n')
        for i in range(len(output)):
            f.write(description[i]+'\t'+ground[i]+'\t'+output[i])
            if i!=len(output)-1:
                f.write('\n')
    logging.info("done!!!")

def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, clamp='clamp',split = 'test',
                        #  model_path='../../correction_checkpoints/PLAIN_ema_0.9999_200000.pt',
                         model_path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/correction_checkpoints/PLAIN_ema_0.9999_070000.pt',
                         use_ddim=False,
                         ml=256,
                         batch_size =64,num_samples=3300,top_p =1.0,out_dir='generation_outputs',
                        #  outputdir='../../tempregeneratebad.txt'
                         outputdir='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/tempregeneratebad.txt'
                         )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
