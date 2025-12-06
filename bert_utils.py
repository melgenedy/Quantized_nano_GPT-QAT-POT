
import os


# Redirect Hugging Face caches to a larger disk
os.environ["HF_HOME"] = "/local/mnt/workspace/melgened/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/melgened/hf_home/transformers"
os.environ["HF_DATASETS_CACHE"] = "/local/mnt/workspace/melgened/hf_home/datasets"

# Ensure directories exist
for p in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_DATASETS_CACHE"]]:
    os.makedirs(p, exist_ok=True)

import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import copy

import functools

from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.utils import _get_module_name_filter
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
    propagate_annotation,
    QuantizationConfig,
)
if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
    from torch.fx import Node

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization import MovingAverageMinMaxObserver
from torch.ao.quantization.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,get_symmetric_quantization_config,)
# 
from torch.ao.quantization import move_exported_model_to_train, move_exported_model_to_eval
from fakequant_pot import FakeQuantize_PoT

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_testing(model_test):
    out = {}
    #model.eval()
    move_exported_model_to_eval(model_test)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                #logits, loss = model(X, Y)
                logits = model_test(X)
                logits_=(logits[0]).squeeze(1)
                targets_last = Y[:, -1]
                loss = torch.nn.functional.cross_entropy(logits_, targets_last)
            losses[k] = loss.item()
        out[split] = losses.mean()
    #model.train()
    move_exported_model_to_train(model_test)
    return out
##########################################################################


import tiktoken

# ok let's assume gpt-2 encodings by default
#print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(tokens=l)


def generate(Model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= 64 else idx[:, -64:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = Model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        
        import torch.nn.functional as F
        
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


device_type = 'cpu'
device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

################################################################################################

def get_batch_for_bert_test(split,data_dir):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    batch_size_test = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size_test = 128 #1024    

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size_test, (batch_size_test,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size_test]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size_test]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

#################################################################################################


import evaluate

bertscore = evaluate.load("bertscore")

def compute_bert(refs, cands):
    results = bertscore.compute(
        predictions=cands,
        references=refs,
        lang="en",                  # set language
        model_type="roberta-base", # optional; default is roberta-large
        rescale_with_baseline=True
    )
    return results

def compute_bert_compare(model_base, mod_quant, data_dir):

    max_new_tokens = 64 #500 # number of tokens generated in each sample
    temperature = 0.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    n_iters = 10
    baseline_avgF1 = np.zeros(10)
    qat_convert_avgF1 = np.zeros(10)
    qat_latest_avgF1 = np.zeros(10)
    ptq_avgF1 = np.zeros(10)

    with torch.no_grad():

        model_base.eval()
        #move_exported_model_to_eval(model)
        #move_exported_model_to_eval(mod_q)
        move_exported_model_to_eval(mod_quant)
        #move_exported_model_to_eval(m_quant)        
        with ctx:
            for k in range(n_iters):

                X_test, y = get_batch_for_bert_test('val',data_dir)
                X = X_test[:,:64]
                X_ref = X_test[:,64:]
                
                print(decode(X[0].tolist()),end="")
                y_org = model_base.generate(X, max_new_tokens, temperature=temperature, top_k=top_k)
                #print(decode(y_org[0].tolist()),end="")
                #print('\n---------------')

                y_convert = generate(mod_quant, X, max_new_tokens, temperature=temperature, top_k=top_k)
                #print(decode(y_convert[0].tolist()),end="")
                #print('\n---------------')            

                
                # Calculate BLEU scores
                reference_texts = [decode(X_test[0,64:64+64].tolist())]
                baseline_outputs = [decode(y_org[0,64:64+64].tolist())]
                qat_convert_outputs = [decode(y_convert[0,64:64+64].tolist())]

                results = compute_bert(reference_texts, baseline_outputs)

                #print("Per-sentence F1:", [round(f, 4) for f in results["f1"]])
                print("Average F1 - Base:", sum(results["f1"]) / len(results["f1"]))

                baseline_avgF1[k] = sum(results["f1"]) / len(results["f1"])

                results = compute_bert(reference_texts, qat_convert_outputs)

                #print("Per-sentence F1:", [round(f, 4) for f in results["f1"]])
                print("Average F1 - qat convert:", sum(results["f1"]) / len(results["f1"]))

                qat_convert_avgF1[k] = sum(results["f1"]) / len(results["f1"])


        model_base.train()
        move_exported_model_to_train(mod_quant)

        return baseline_avgF1.mean(), qat_convert_avgF1.mean()
