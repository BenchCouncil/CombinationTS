from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
from torch import nn
# from Wrap import WrapEmbedding, WrapEncoder, WrapDecoder
from .Block import Block
from utils.tools import is_list
from layers.normalization import MultiStreamNorm, Normalize
from layers.decompsition import TSDecomp
from layers.filter import GraphFilter, build_filter_mask
from layers.time_embedding import TemporalEmbedding, TimeFeatureEmbedding
from arch import BaseModule

class Model(nn.Module):
    def __init__(self, cfg: DictConfig, device: torch.device=torch.device("cuda")):
        super(Model, self).__init__()
        self.device = device
        self.cache = {}
        self.shapes = []
        input_shape = (cfg.n_vars, cfg.seq_len, 1)
        self.shapes.append(input_shape)
        share_cfg = cfg.get("share", {'embedding': False, 'encoder': False, 'decoder': False})
        share_embedding, share_encoder, share_decoder = \
            share_cfg.get("embedding", False), share_cfg.get("encoder", False), share_cfg.get("decoder", False)
        self.use_norm, self.use_multi_stream, self.multi_module_cfg = cfg.use_norm, cfg.use_multi_stream, cfg.use_multi_module_cfg
        self.use_cycle = cfg.get("use_cycle", False)
        self.channel_independence = cfg.get("channel_independence", False)
        self.use_post_decomp = cfg.get("post_decomp", False) and self.use_multi_stream
        self.use_filter = cfg.get("use_filter", False)
        self.use_time_feature = cfg.get("use_time_feature", False)
        embedding_cfg = cfg.embedding if is_list(cfg.embedding) else [cfg.embedding]
        encoder_cfg = cfg.encoder if is_list(cfg.encoder) else [cfg.encoder]
        decoder_cfg = cfg.decoder if is_list(cfg.decoder) else [cfg.decoder]

        # Parameters to build modules
        params_dict = {
            "seq_len": cfg.seq_len,
            "label_len": cfg.label_len,
            "pred_len": cfg.pred_len,
            "n_vars": cfg.n_vars,
            "d_model": cfg.d_model,
            "dropout": cfg.dropout,
            "channel_independence": cfg.channel_independence,
        }

        self.denorm_after_stack = True
        # Multi-stream decomposition
        if self.use_multi_stream:
            self.denorm_after_stack = cfg.multi_stream.get("denorm_after_stack", True)
            self.multi_stream_decomp = Block(cfg.multi_stream, self.shapes[-1], **params_dict)
            self.shapes.append(self.multi_stream_decomp.output_shapes)

        # Normalization
        if self.use_norm:
            if self.use_multi_stream:
                self.norm = MultiStreamNorm(input_shape=self.shapes[-1], use_multi_stream=True)
            else:
                self.norm = Normalize()
        else:
            self.norm = None

        if self.use_cycle:
            assert not self.use_multi_stream, "Multi-stream decomposition and cycle index has not been tested together."
            self.cycle = instantiate(cfg.cycle)
            
        # Embedding
        self.embedding = Block(embedding_cfg, self.shapes[-1], shared_module=share_embedding, **params_dict)
        self.shapes.append(self.embedding.output_shapes)

        # Length expansion trick from TimesNet
        self.use_length_expansion = cfg.get("use_length_expansion", False)
        if self.use_length_expansion:
            self.length_expansion = nn.ModuleList()
            expansion_shape = []
            for C, L, D in self.shapes[-1]:
                assert L != 1, "Length expansion is not needed when sequence length is 1"
                target_L = (cfg.pred_len + cfg.seq_len) / cfg.seq_len * L
                self.length_expansion.append(nn.Linear(L, int(target_L)))
                expansion_shape.append((C, int(target_L), D))
            self.shapes.append(expansion_shape)
            def length_expansion_hook(module, input, output):
                for i in range(len(output)):
                    output[i] = output[i].permute(0, 1, 3, 2)  # (B, C, L, D) -> (B, C, D, L)
                    output[i] = self.length_expansion[i](output[i])  # (B, C, D, L) -> (B, C, D, L_new)
                    output[i] = output[i].permute(0, 1, 3, 2)  # (B, C, D, L_new) -> (B, C, L_new, D)
                return output
            self.embedding.register_forward_hook(length_expansion_hook)

        # Encoder
        self.encoder = Block(encoder_cfg, self.shapes[-1], shared_module=share_encoder, **params_dict)
        self.shapes.append(self.encoder.output_shapes)

        # Decoder
        decoder_params_dict = params_dict.copy()    # make a copy to avoid modification
        if self.use_length_expansion:
            # modify pred_len for decoder when using length expansion
            decoder_params_dict['seq_len'] = int(cfg.seq_len + cfg.pred_len)
        self.decoder = Block(decoder_cfg, self.shapes[-1], shared_module=share_decoder, **decoder_params_dict)
        self.shapes.append(self.decoder.output_shapes)

        print("Model shapes:", self.shapes)

        # Change Model input/output format: (B, L, N) -> (B, N, L, 1)
        # to keep the pattern (B, C, L, D) in embedding/encoder/decoder blocks
        def forward_pre_hook(module, input):
            input = list(input)
            x, x_mark = input[0], input[1]
            x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, L, N) -> (B, N, L, 1)
            x_mark = x_mark.permute(0, 2, 1).unsqueeze(-1) if x_mark is not None else None
            input[0], input[1] = x, x_mark
            return tuple(input)
        self.register_forward_pre_hook(forward_pre_hook)
        def forward_post_hook(module, input, output):
            if isinstance(output, list):
                output = output[0]
            output = output.squeeze(-1).permute(0, 2, 1)  # (B, N, L, 1) -> (B, L, N)
            return output
        self.register_forward_hook(forward_post_hook)

        # Post Decomposition hooks for TimeMixer
        if self.use_post_decomp:
            self._register_time_mixer_hooks(cfg)

        # Filter hooks for GraphFilter. Add GraphFilter in each encoder layer
        if self.use_filter:
            self._register_filter_hooks(cfg)
        
        # Time Feature hooks to add time feature to embedding output
        if self.use_time_feature:
            self._register_time_feature_hooks(cfg)


    def forward(self, x, x_mark, x_dec=None, x_mark_dec=None, batch_cycle=None): 
        x, x_mark = self.multi_stream_decomp(x, x_mark) if self.use_multi_stream else (x, x_mark)
        x = self.norm(x) if self.norm else x
        x = self.cycle(x, batch_cycle) if self.use_cycle else x
        x = self.embedding(x, x_mark)
        x = self.encoder(x)
        x = self.decoder(x)

        if self.denorm_after_stack and is_list(x) and len(x) > 1:
            x = torch.stack(x, dim=-1).sum(-1)

        x = self.norm(x, 'denorm') if self.norm else x

        if not self.denorm_after_stack and is_list(x) and len(x) > 1:
            x = torch.stack(x, dim=-1).sum(-1)

        x = self.cycle(x, batch_cycle, type='forecast') if self.use_cycle else x
        return x


    def _register_time_mixer_hooks(self, cfg: DictConfig):
        assert self.use_post_decomp, "post_decomp should be True to register TimeMixer hooks"
        self.ts_decomp = TSDecomp(kernel_size=cfg.get("post_decomp_kernel_size", 25))
        def post_decomp_hook(module, input, output: tuple[list[torch.Tensor], list[torch.Tensor]]):
            if self.channel_independence:
                return output
            else:
                res = list(output)
                x_list = output[0]
                trend_list = []
                seasonal_list = []
                for x in x_list:
                    seasonal, trend = self.ts_decomp(x.squeeze(-1).permute(0,2,1))
                    seasonal_list.append(seasonal.permute(0,2,1).unsqueeze(-1))
                    trend_list.append(trend.permute(0,2,1).unsqueeze(-1))
                res[0] = seasonal_list
                self.cache['trend_list'] = trend_list
            return tuple(res)
        self.multi_stream_decomp.register_forward_hook(post_decomp_hook)
        def fmm_pre_hook(module, input):
            if self.channel_independence:
                trend_list = self.cache.get('trend_list', None)
                input = list(input)
                input.append(trend_list)
                return tuple(input)
            else: return input
        self.decoder.register_forward_pre_hook(fmm_pre_hook)
    
    def _register_filter_hooks(self, cfg: DictConfig):
        assert cfg.get("use_filter", False), "use_filter should be True to register filter hooks"
        filter_cfg = cfg['filter']
        enc_in_shapes = self.encoder.init_shapes
        e_layers = cfg.encoder['e_layers']
        for enc in self.encoder.block_modules:
            assert isinstance(enc, BaseModule), "Encoder block module is not an instance of BaseModule"
            assert len(enc.init_shape) == 1, "GraphFilter currently only supports single input tensor"
            assert len(enc.layers) == e_layers, "Encoder layers do not match e_layers, \
                please assign the repeated encoder layers to `self.layers` in Encoder module"
            input_shape = enc.init_shape[0]
            mask = build_filter_mask(input_shape, self.device)
            C, L, D = input_shape
            for layer in enc.layers:
                in_dim = C * L
                filter = GraphFilter(
                    d_model=D,
                    n_vars=C,
                    n_heads=filter_cfg.get("n_heads", 4),
                    top_p=filter_cfg.get("top_p", 0.5),
                    dropout=cfg['dropout'],
                    in_dim=in_dim
                )
                layer.add_module('graph_filter', filter)
                layer.add_module('norm1', nn.LayerNorm(D))
                layer.add_module('norm2', nn.LayerNorm(D))
                layer.register_buffer('ori_x', torch.zeros(1))  # dummy buffer to avoid error
                def filter_x_hook(module, input):
                    x = input[0]
                    x_shape = x.shape
                    C, L, D = input_shape
                    B = cfg['batch_size']
                    x = x.reshape(-1, C * L, D) #XXX: X should not be permuted before encoder layers
                    x = module.norm1(x)
                    out, _ = module.graph_filter(x, masks=mask, alpha=filter_cfg['alpha'], is_training=module.training)
                    x = x + out
                    module.ori_x = x.reshape(x_shape)  # save for residual connection
                    out_x = module.norm2(x)
                    out_x = out_x.reshape(x_shape)
                    return (out_x,)
                def filter_residual_hook(module, input, output):
                    x = module.ori_x
                    if isinstance(output, tuple):
                        output = list(output)
                        output[0] = output[0] + x
                        return tuple(output)
                    else:
                        return output + x
                layer.register_forward_pre_hook(filter_x_hook)
                layer.register_forward_hook(filter_residual_hook)
                    
    def _register_time_feature_hooks(self, cfg: DictConfig):
        assert self.use_time_feature, "use_time_feature should be True to register TimeFeature hooks"
        time_feature_cfg = cfg['time_feature']
        temporal_embed_type, freq = time_feature_cfg['temporal_embed_type'], time_feature_cfg['freq']
        self.temporal_embedding = TemporalEmbedding(d_model=cfg.d_model, embed_type=temporal_embed_type, freq=freq) if temporal_embed_type != 'timeF' \
            else TimeFeatureEmbedding(d_model=cfg.d_model, embed_type=temporal_embed_type, freq=freq)
        def time_feature_hook(module, input, output):
            x_mark = input[1]
            x_mark_list = x_mark if isinstance(x_mark, list) else [x_mark]
            assert isinstance(output, list), "Output of embedding block should be a list."
            assert len(output) == len(x_mark_list), "code below assumes output and x_mark have the same length."
            for i in range(len(output)):
                time_feature = self.temporal_embedding(x_mark_list[i])
                B, C, L, D = output[i].shape
                Bt, Lt, Dt = time_feature.shape
                assert Lt % L == 0, "Time feature length must be multiple of embedding output length."
                assert B == Bt and D == Dt, "Batch size or d_model dimension does not match between embedding output and time feature."
                stride = Lt // L
                time_feature = time_feature[:, ::stride, :].unsqueeze(1).expand(-1, C, -1, -1)  # (B, L, D) -> (B, 1, L, D) -> (B, C, L, D)
                output[i] = output[i] + time_feature
            return output
        self.embedding.register_forward_hook(time_feature_hook)