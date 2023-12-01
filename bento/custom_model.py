from xbert import BertConfig, BertForMaskedLM
import torch
from torch import nn
import numpy as np
from bisect import bisect_left
from torch.distributions import Categorical
from d_pv2smiles_stochastic import generate_with_property
from calc_property import calculate_property
from d_smiles2pv import pv_generate
from transformers import BertTokenizer, WordpieceTokenizer


def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


@torch.no_grad()
def generate(self, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = self.text_encoder(text,
                                     attention_mask=text_atts,
                                     encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=prop_att_mask,
                                     return_dict=True,
                                     is_decoder=True,
                                     return_logits=True,
                                     )[:, -1, :]  # batch*300
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1


class SPMM_inference(nn.Module):
    def __init__(self, tokenizer=None, config=None, device='cpu'):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        embed_dim = config['embed_dim']

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        property_width = text_width

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width * 2, 2)

        self.property_embed = nn.Linear(1, property_width)
        bert_config2 = BertConfig.from_json_file(config['bert_config_property'])
        self.property_encoder = BertForMaskedLM(config=bert_config2).bert
        self.property_mtr_head = nn.Sequential(nn.Linear(property_width, property_width),
                                               nn.GELU(),
                                               nn.LayerNorm(property_width, bert_config.layer_norm_eps),
                                               nn.Linear(property_width, 1))
        self.property_cls = nn.Parameter(torch.zeros(1, 1, property_width))
        self.property_mask = nn.Parameter(torch.zeros(1, 1, property_width))    # unk token for PV
        self.property_mlm_mask = nn.Parameter(torch.zeros(1, 1, property_width))    # mask token for PV MLM

    def forward(self, mode='p2s', p2s_prop=None, p2s_mask=None, stochastic=True, k=2, n_sample=10, s2p_smiles=None):
        if mode == 'p2s':
            if isinstance(p2s_prop, np.ndarray):
                p2s_prop = torch.from_numpy(p2s_prop)
            if isinstance(p2s_mask, np.ndarray):
                p2s_mask = torch.from_numpy(p2s_mask)
            p2s_prop = p2s_prop.to(self.device)
            p2s_mask = p2s_mask.to(self.device).long()
            smiles = generate_with_property(self, properties=p2s_prop, n_sample=n_sample, prop_mask=p2s_mask, k=k, stochastic=stochastic)
            return smiles
        elif mode == 's2p':
            if not isinstance(s2p_smiles, list):
                s2p_smiles = [s2p_smiles]
            pvs = pv_generate(self, s2p_smiles)     # list of tensor[1*53]
            return torch.cat(pvs, dim=0).numpy()
        else:
            raise ValueError(f'Unknown mode: {mode}')


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
    model = SPMM_inference(config=config, tokenizer=tokenizer, device=device)
    model.eval()

    print('LOADING PRETRAINED MODEL..')
    checkpoint = torch.load(ckpt_name, map_location='cpu')
    state_dict = checkpoint['state_dict']

    for key in list(state_dict.keys()):
        if 'word_embeddings' in key and 'property_encoder' in key:
            del state_dict[key]
        elif 'queue' in key or '_m.' in key:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % ckpt_name)
    print(msg)
    model = model.to(torch.device(device))
    return model


if __name__ == '__main__':
    model = get_model("./Pretrain/checkpoint_SPMM_20m.ckpt", device='cuda')

    print(model(mode='p2s', p2s_prop=calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1'), p2s_mask=torch.zeros(53), stochastic=True, k=2, n_sample=50))
    print("="*100)
    print(model(mode='s2p', s2p_smiles=['COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1']*64))
    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
