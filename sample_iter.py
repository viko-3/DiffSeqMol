import argparse
import os, json
import pickle
from tracemalloc import start
import os.path as op
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
import pandas as pd
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rdkit import rdBase
from tdc import Oracle

rdBase.DisableLog('rdApp.error')
import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import Chem, DataStructs
from utils import sascorer
import networkx as nx
import shutil, os
import rdkit.Chem.QED as QED

raw_data = []


def similarity(a, b, sim_type="binary"):
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    if sim_type == "binary":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    else:
        fp1 = AllChem.GetMorganFingerprint(amol, 2, useChirality=False)
        fp2 = AllChem.GetMorganFingerprint(bmol, 2, useChirality=False)

    sim = DataStructs.TanimotoSimilarity(fp1, fp2)

    return sim


def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def qed(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return 0.0
    return QED.qed(mol)


oracle = Oracle(name='DRD2')


def DRD2_score(smi):
    return oracle(smi)


def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main(num, sample_num=20):
    global raw_data
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    #set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join("datasets/test/samples.jsonl")
    # fout = open(out_path, 'a')

    all_test_data = []

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')

    from tqdm import tqdm

    for index, cond in enumerate(all_test_data):

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        x_start = x_start.repeat(sample_num, 1, 1)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb_copy.cuda()),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )

        model_emb_copy.cpu()
        # print(samples[0].shape) # samples for each step

        sample = samples[-1]
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()
        # print('decoding for seq2seq', )
        # print(arr.shape)

        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices
        # tokenizer = load_tokenizer(args)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori.repeat(sample_num, 1)):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))

            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        if num == 1:
            raw_data.extend(
                [temp.replace('</s>', '').replace('<s>', '').replace("[CLS]", '').replace("[SEP]", '') for temp
                 in word_lst_source])

        df = pd.DataFrame()
        source, recover, reference = [], [], []
        # gen->recov,ref  real->src
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref * sample_num, word_lst_source * sample_num):
            source.append(
                src.replace('</s>', '').replace('<s>', '').replace("[CLS]", '').replace("[SEP]", ''))
            recover.append(
                recov.replace('</s>', '').replace('<s>', '').replace("[CLS]", '').replace("[SEP]", ''))

        df['source'], df['recover'] = source, recover
        reference = raw_data[index * args.batch_size:(index + 1) * args.batch_size]
        df['reference'] = reference * sample_num

        if num == 5:
            df = filter(df, caculate_div=True)
        else:
            df = filter(df, caculate_div=False)
        if df is None:
            df = pd.DataFrame()
            df["recover"], df["reference"], df["source"] = [source[0]], reference, [source[0]]
        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(df["recover"], df["reference"], df["source"]):
            print(json.dumps({"trg": recov, "src": src, "ref": ref}), file=fout)
        fout.close()

    shutil.copy(out_path, f'{args.data_dir}/{args.split}{num}.jsonl')
    shutil.move(out_path, f'{args.data_dir}/{args.split}.jsonl')

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


valid_sum = 0
oracle_DRD2 = Oracle(name='DRD2')
oracle_QED = Oracle(name='QED')
from tdc import Evaluator

evaluator = Evaluator(name='Diversity')


def filter(df, caculate_div=False):
    global div

    def compare_plogp(source, recover, similarity):
        if similarity < 0.4:
            return None
        source_plogp = penalized_logp(source)
        recover_plogp = penalized_logp(recover)
 
        if recover_plogp > source_plogp:
            return recover
        else:
            return None

    def compare_DRD2(source, recover, similarity):
        if similarity < 0.4:
            return None
        source_score = oracle_DRD2(source)
        recover_score = oracle_DRD2(recover)
        if caculate_div:
            if recover_score >= 0.5:
                return recover
        else:
            if recover_score > source_score:
                return recover
        return None

    def compare_QED(source, recover, similarity):
        if similarity < 0.4:
            return None
        source_QED = oracle_QED(source)
        recover_QED = oracle_QED(recover)
        if caculate_div:
            if recover_QED >= 0.9:
                return recover
        else:
            if recover_QED > source_QED:
                return recover
        return None

    # logp
    
    df['recover_plogp'] = df['recover'].apply(penalized_logp)
    df['similarity'] = df.apply(lambda x: similarity(x['reference'], x['recover']), axis=1)
    df['source'] = df.apply(lambda x: compare_plogp(x['source'], x['recover'], x['similarity']), axis=1)
    # 
    df = df.dropna()
    if len(df) != 0:
        if caculate_div:
            smi_set = set(df['recover'].values)
            if len(smi_set) == 1:
                diversity = 0
            else:
                diversity = evaluator(smi_set)
            div.append(diversity)
            print(np.nanmean(div))
        df = df.sort_values(by=['recover_plogp']).iloc[-1:]
        return df
    """
    # DRD2
    df['recover_score'] = df['recover'].apply(oracle_DRD2)
    df['similarity'] = df.apply(lambda x: similarity(x['reference'], x['recover']), axis=1)
    df['source'] = df.apply(lambda x: compare_DRD2(x['source'], x['recover'], x['similarity']), axis=1)
    df = df.dropna()
    if len(df) != 0:
        if caculate_div:
            smi_set = set(df['recover'].values)
            if len(smi_set) == 1:
                diversity = 0
            else:
                diversity = evaluator(smi_set)
            div.append(diversity)
            print(np.nanmean(div))
        df = df.sort_values(by=['recover_score']).iloc[-1:]
        return df
    
    # QED
    df['recover_QED'] = df['recover'].apply(penalized_logp)
    df['similarity'] = df.apply(lambda x: similarity(x['reference'], x['recover']), axis=1)
    df['source'] = df.apply(lambda x: compare_DRD2(x['source'], x['recover'], x['similarity']), axis=1)
    df = df.dropna()
    if len(df) != 0:
        if caculate_div:
            smi_set = set(df['recover'].values)
            if len(smi_set) == 1:
                diversity = 0
            else:
                diversity = evaluator(smi_set)
            div.append(diversity)
            print(np.nanmean(div))
        df = df.sort_values(by=['recover_score']).iloc[-1:]
        return df"""
    return None


if __name__ == '__main__':
    div = []
    iter_num = 5
    for i in range(iter_num):
        main(i + 1)
        print(np.nanmean(div))
