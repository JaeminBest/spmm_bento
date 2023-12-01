"""
inference.py

DEMO inference script for VESSL. This script is used to run inference on a single image or a directory of images.
this demo is only for Medical X-VL dataset and models.
"""
from SPMM_models import SPMM
from calc_property import calculate_property
from d_pv2smiles_stochastic import generate_with_property
from d_smiles2pv import pv_generate
import torch
from transformers import BertTokenizer, WordpieceTokenizer


def get_model(ckpt_name, device='cuda'):
    config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 8,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size': 32768,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': 1e-4, 'epochs': 30, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 1e-4, 'weight_decay': 0.02}
    }
    tokenizer = BertTokenizer(vocab_file='./vocab_bpe_300.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)

    print('LOADING PRETRAINED MODEL..')
    checkpoint = torch.load(ckpt_name, map_location='cpu')
    state_dict = checkpoint['state_dict']

    for key in list(state_dict.keys()):
        if 'word_embeddings' in key and 'property_encoder' in key:
            del state_dict[key]
        if 'queue' in key:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % ckpt_name)
    print(msg)
    model = model.to(torch.device(device))
    return model


# pv_to_smiles
spmm_pv_to_smiles_model = get_model("./Pretrain/checkpoint_SPMM_20m.ckpt", device='cuda')

test_input_mask1 = torch.zeros(53)         # 0 indicates no masking for that property
test_input_prop1 = calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1')
output = generate_with_property(spmm_pv_to_smiles_model, properties=test_input_prop1, n_sample=10, prop_mask=test_input_mask1, stochastic=True)
print(output)

# smiles_to_pv
spmm_smiles_to_pv_model = get_model("./Pretrain/checkpoint_SPMM_20m.ckpt", device='cuda')
output = pv_generate(spmm_smiles_to_pv_model, ['COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1', 'c1ccccc1'])
print(len(output), output[0].size())
