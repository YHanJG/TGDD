"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import selfies as sf
import record
import logging
from mytokenizers import SimpleSmilesTokenizer
import argparse
import os, json
from rdkit import Chem
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

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from mydatasets import get_dataloader,ChEBIdataset

def main():
    args = create_argparser().parse_args()
    set_seed(121)

    # dist_util.setup_dist()
    logger.configure()
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    from mytokenizers import regexTokenizer,selfTokenizer
    if args.is_self:
        tokenizer = selfTokenizer()
    else:
        tokenizer = regexTokenizer()
    # model = TransformerNetModel2(
    #     in_channels=32,  # 3, DEBUG**
    #     # deep_channels = 10,
    #     model_channels=128,
    #     dropout=0.1,
    #     use_checkpoint=False,
    #     config_name='bert-base-uncased',
    #     training_mode='e2e',
    #     vocab_size=len(tokenizer),
    #     experiment_mode='lm',
    #     logits_mode=1,
    #     hidden_size = 1024,
    #     num_attention_heads=16,
    #     num_hidden_layers = 12,
    # )
    model = TransformerNetModel2(
        in_channels=32,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(tokenizer),
        experiment_mode='lm',
        logits_mode=1,
        is_join=args.is_join,
        hidden_size = 1024,
        num_attention_heads=16,
        num_hidden_layers = 3,
    )
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0,2000,10)],
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
    logger.log(f'the parameter count is {pytorch_total_params}')

    # diffusion.rescale_timesteps = False  # DEBUG --> REMOVE
    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval() # DEBUG

    logger.log("sampling...")
    print(args.num_samples)
    # model3 = get_weights(model2, args)
    print('--'*30)
    print('loading {} set'.format(args.split))
    print('--'*30)
    dir = args.dataser_dir
    train_dataset = ChEBIdataset(
        dir=dir,
        is_self = args.is_self,
        smi_tokenizer=tokenizer,
        self_tokenizer=tokenizer,
        split=args.split,
        replace_desc=False
        # pre = pre
    )
    print('DATASETINFO-----------------------------')
    print(len(train_dataset),(train_dataset[0]['desc_state'].shape))
    if not args.is_self:
        desc = [(train_dataset[i]['desc_state'],train_dataset[i]['desc_mask'],train_dataset[i]['smiles']) for i in range(args.num_samples)]
    else:
        desc = [(train_dataset[i]['desc_state'],train_dataset[i]['desc_mask'],train_dataset[i]['smiles'],train_dataset[i]['selfies']) for i in range(args.num_samples)]
    answer = [i[2] for i in desc]
    model3 = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model3.requires_grad = False
    if not args.ev:
        allsample = []
        num_done = 0
        while num_done < args.num_samples:
            idend = min(num_done+args.batch_size,args.num_samples)
            print('acquiring  {} : {}'.format(num_done,idend))
            desc_state = th.concat([i[0] for i in desc[num_done:idend]],dim=0)
            desc_mask = th.concat([i[1] for i in desc[num_done:idend]],dim=0)
            
            model_kwargs = {}
            print('use_ddim:{}',args.use_ddim)
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample_shape = (idend-num_done, 256, model.in_channels)
            sample = sample_fn(
                model,
                sample_shape,
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
        th.save(sample, args.save_path)
        print('采样结果已存储到文件中')
    else:
        sample = th.load(args.load_path)
    print("sample:",sample.shape)
    print('decoding for e2e', )
    print(sample.shape)#sample: torch.Size([3300, 256, 32])
    # x_t = th.tensor(sample).cuda()
    x_t = sample.clone().detach().cuda()
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    print(logits)
    print("logits",logits.shape)#logits torch.Size([3300, 256, 257])
    cands,sample = th.topk(logits, k=1, dim=-1)
    print(cands)
    print(sample)
    # sample = cands.indices
    print("sample:",sample.shape)#torch.Size([3300, 256, 1])
    print(sample)
    sample = sample.squeeze(-1)
    print(sample)
    if args.is_self:
        tokenizer = selfTokenizer()
    else:
        tokenizer = regexTokenizer()
    c = tokenizer.decode(sample)
    print(c[0])
    if not args.is_self:
        with open(args.outputdir,'w') as f:
            for i,x in enumerate(c):
                if i==0:
                    print(x)
                f.write(x.replace('[PAD]','')+'   ||   '+answer[i]+'\n')
    else:
        with open(args.outputdir,'w') as f:
            for i,x in enumerate(c):
                if i==0:
                    print(x)
                f.write(x+'   ||   '+(answer[i])+'\n')

    # with open(args.outputdir) as f:
    #     allsmiles = [(k.strip().split('||')[0].strip().replace('[EOS]','').replace('[SOS]','')) for k in f.readlines()]
    # f = open(args.tempbadmols,'w')
    # for cnt,s in enumerate(allsmiles):
    #     mol = Chem.MolFromSmiles(s)
    #     if mol is None:
    #         f.write(str(cnt)+'\t'+s+'\n')
    f.close()
    print("done!!!")
    # logger = logging.getLogger()
    # # 移除之前的处理程序（如果存在）
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    logging.info("ev...")
    logging.info(f'checkpoint: {args.model_path}')
    if not args.is_self and not args.is_join:
        gt,op = get_smis('/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/textguidtry_256_final.txt')
    else:
        gt,op = get_smis(args.outputdir)
    bleu_score, exact_match_score, levenshtein_score,_  = evaluate(gt,op)
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fevaluate(gt,op)
    logging.info(f'BLEU: {round(bleu_score, 3)}')
    logging.info(f'Exact: {round(exact_match_score, 3)}')
    logging.info(f'Levenshtein: {round(levenshtein_score, 3)}')
    logging.info(f'MACCS FTS: {round(maccs_sims_score, 3)}')
    logging.info(f'RDK FTS: {round(rdk_sims_score, 3)}')
    logging.info(f'Morgan FTS: {round(morgan_sims_score, 3)}')
    logging.info(f'Validity: {round(validity_score, 3)}')

    try:
        fcd_metric_score = fcdevaluate(gt, op)
        logging.info(f'FCD Metric: {round(fcd_metric_score, 3)}')
        print(f'FCD Metric: {round(fcd_metric_score, 3)}')
    except Exception as e:
        logging.info(f'FCD Metric: FAIL!!')
        print(f'FCD Metric: FAIL!!')
    logging.info(f'Text2Mol: miss')

    print(f'BLEU: {round(bleu_score, 3)}')
    print(f'Exact: {round(exact_match_score, 3)}')
    print(f'Levenshtein: {round(levenshtein_score, 3)}')
    print(f'MACCS FTS: {round(maccs_sims_score, 3)}')
    print(f'RDK FTS: {round(rdk_sims_score, 3)}')
    print(f'Morgan FTS: {round(morgan_sims_score, 3)}')
    print(f'Validity: {round(validity_score, 3)}')
    print(f'Text2Mol: miss')

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
                         model_path='/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/check_join/nojoin_test_3/1.9999999999999995e-05_20066206_PLAIN_ema_0.9999_200000.pt',
                         use_ddim=False,
                         batch_size =64,num_samples=3300,top_p =1.0,out_dir='generation_outputs_not_jointest_20',
                         outputdir='/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/check_join/nojoin_test_3/textguidtry_256_not_jointest_20.txt',
                         is_self=False,
                         dataser_dir = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/datasets/SMILES/',
                         tempbadmols = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/check_join/nojoin_test_3/tempbadmols_20.txt',
                         save_path = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/check_join/nojoin_test_3/not_jointest_20.pt',
                         load_path = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/check_join/nojoin_test_3/not_jointest_20.pt',
                         is_join=False,
                         ev=False,
                         )
    if text_defaults['is_self']:
        text_defaults['out_dir'] = 'generation_outputs_self'
        text_defaults['tempbadmols'] = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/tempbadmols_self.txt'
        text_defaults['model_path'] = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/checkpoints_self/5.000000000032757e-10_20112382_PLAIN_ema_0.9999_200000.pt'
        text_defaults['outputdir'] = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/20112382_textguidtry_256_final_self_20.txt'
        text_defaults['dataser_dir'] = '/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/datasets/SELFIES/'
        model_path = text_defaults['model_path']
        last_part = model_path.split('/')[-1]
        desired_model = 'text_result'+last_part.replace('.pt','').split('_')[-1]+'.pt'
        text_defaults['save_path'] = os.path.join('/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/',desired_model)
        text_defaults['load_path'] = text_defaults['save_path']
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    if text_defaults['is_self']:
        filename = '/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/improved-diffusion/log/text_sample_self.log'
    else:
         filename='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/improved-diffusion/log/text_sample_14.log'
    logging.basicConfig(filename=filename, level=logging.INFO)
    logging.info("开始text_sample...")
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

import numpy as np
import os.path as osp
from nltk.translate.bleu_score import corpus_bleu
from rdkit import RDLogger
from Levenshtein import distance as lev
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')
from fcd import get_fcd, load_ref_model, canonical_smiles
import warnings
warnings.filterwarnings('ignore')
def get_smis(filepath):
    print(filepath)
    with open(filepath) as f:
        lines = f.readlines()
    gt_smis= []
    op_smis = []
    for s in lines:
        if len(s)<3:
            continue
        s0,s1 = s.split(' || ')
        s0,s1 = s0.strip().replace('[EOS]','').replace('[SOS]','').replace('[X]','').replace('[XPara]','').replace('[XRing]',''),s1.strip()
        gt_smis.append(s1)
        op_smis.append(s0)
    return gt_smis,op_smis


def evaluate(gt_smis,op_smis):
    references = []
    hypotheses = []
    for i, (gt, out) in enumerate(zip(gt_smis,op_smis)):
        gt_tokens = [c for c in gt]
        out_tokens = [c for c in out]
        references.append([gt_tokens])
        hypotheses.append(out_tokens)
    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    references = []
    hypotheses = []
    levs = []
    num_exact = 0
    bad_mols = 0
    for i, (gt, out) in enumerate(zip(gt_smis,op_smis)):
        hypotheses.append(out)
        references.append(gt)
        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
        except:
            bad_mols += 1
        levs.append(lev(out, gt))
    # Exact matching score
    exact_match_score = num_exact/(i+1)
    # Levenshtein score
    levenshtein_score = np.mean(levs)
    validity_score = 1 - bad_mols/len(gt_smis)
    return bleu_score, exact_match_score, levenshtein_score, validity_score


def fevaluate(gt_smis,op_smis, morgan_r=2):
    outputs = []
    bad_mols = 0
    for n, (gt_smi,ot_smi) in enumerate(zip(gt_smis,op_smis)):
        try:
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)
            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append((gt_m, ot_m))
        except:
            bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    enum_list = outputs
    for i, (gt_m, ot_m) in enumerate(enum_list):
        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score

def fcdevaluate(qgt_smis,qop_smis):
    gt_smis = []
    ot_smis = []
    for n, (gt_smi,ot_smi) in enumerate(zip(qgt_smis,qop_smis)):
        if len(ot_smi) == 0: ot_smi = '[]'
        gt_smis.append(gt_smi)
        ot_smis.append(ot_smi)
    model = load_ref_model()
    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]
    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    return fcd_sim_score

# gt,op = get_smis('tempoutput.txt')
    

if __name__ == "__main__":
    # logging.basicConfig(filename='/home/jianghanyuhd/code/tgm-dlm-main/tgm-dlm-main/improved-diffusion/scripts/text_sample.log', level=logging.INFO)
    # logging.info("开始text_sample...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
    record.record()
