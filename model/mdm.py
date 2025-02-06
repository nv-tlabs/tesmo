import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.positional_encoding import PositionalEncoding, TimestepEmbedder
from model.tools import OutputProcess, InputProcess, EmbedAction

### ! MDM works in the normalized motion space; While we keep scene condition in real space;

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, inv_transform_th=None, **kargs):
        super().__init__()

        # ! inv_transform_th is the function from data loader.

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        # import pdb;pdb.set_trace()
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        
        
        self.cfg_startEnd = kargs.get('cfg_startEnd', False)
        # self.cfg_startEnd_addNoise = kargs.get('cfg_startEnd_addNoise', False)
        self.cond_mask_prob_startEnd = kargs.get('cond_mask_prob_startEnd', 0.)
        print(f'cfg on start and end: {self.cfg_startEnd}; scale: {self.cond_mask_prob_startEnd}')
        # print(f'cfg on addNoise on the start and end: {self.cfg_startEnd_addNoise}; scale: {self.cond_mask_prob_startEnd}')
        

        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout) # dropout the text in the input.
        self.emb_trans_dec = emb_trans_dec

        # import pdb;pdb.set_trace()
        if self.arch == 'trans_enc' or self.arch == 'trans_enc_new_attention': # this is US.
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        # TODO: get the original position before normalization.  
        # self.dataset_loader = dataset_loader # .inv_transform_th()
        self.inv_transform_th = inv_transform_th
        # self.recover_from_ric = 
        # data representation;
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False, return_mask=False):
        # import pdb;pdb.set_trace()
        bs, d = cond.shape
        if force_mask:
            if return_mask:
                return torch.zeros_like(cond), torch.zeros(bs, dtype=bool)
            else:
                return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.: # drop condition
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            if return_mask:
                cond_mask = ~(cond == 0.0).all(axis=1)
                cond_mask = (1. - mask) & cond_mask
                return cond * (1. - mask), cond_mask    
            return cond * (1. - mask)
        else:
            if return_mask:
                cond_mask = ~(cond == 0.0).all(axis=1)
                return cond, cond_mask
            return cond


    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        
        
        #### CFG for text and action embedding
        force_mask = y.get('uncond', False)
        
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)) # timestep embed + text embed

        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        
        
        # cfg_motion_control=y.get('uncond_motion_control', False) # this is useless.
        # cfg_motion_control_addNoise=y.get('uncond_motion_control_addNoise', False)
        

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_enc_new_attention':
            # add the mask for the input start and end point.
            if self.cfg_startEnd:
                import pdb;pdb.set_trace()
                startEnd_mask = torch.bernoulli(torch.ones(bs, device=y['mask'].device) * self.cond_mask_prob_startEnd).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
                for b_i in range(bs):
                    last_frame = y['lengths'][b_i]
                    y['mask'][b_i, :, :, [0, last_frame-1]] *= (1-startEnd_mask[b_i]).bool()
                    
            # if cfg_motion_control: # used for inference.
            #     for b_i in range(bs):
            #         last_frame = y['lengths'][b_i]
            #         y['mask'][b_i, :, :, [0, last_frame-1]] *= 0.0
                    
            x_mask = y['mask']
            x_mask = x_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, -1)
            time_mask = torch.ones(bs, dtype=bool).reshape(1, bs, -1).to(x_mask.device)
            
            
            
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            aug_mask = torch.cat((time_mask, x_mask), 0).to(xseq.device) # ignore the attention for the padding elements
            # import pdb;pdb.set_trace()
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask[:,:,0].permute(1,0))[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
    
            
        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


## fix up the bugs in transformer.
## TODO;





