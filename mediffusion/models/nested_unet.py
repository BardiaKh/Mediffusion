import torch
import torch.nn as nn
import inspect
import math
from copy import copy

from mediffusion.modules.nn import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding
)

from mediffusion.modules.res import (
    TimestepEmbedSequential,
    Upsample,
    Downsample,
    ResBlock
)
from mediffusion.modules.attention import (
    AttentionBlock,
)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param concat_channels: if specified (as an int), then this model will
        concatenate the concat vector to the input.
    :param guidance_drop_prob: if > 0, then then use classifier-free
        guidance with the given scale.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param scale_skip_connection: use skip connection scaling in res blocks.
    """

    def __init__(
        self,
        input_size, # just for compatibility
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        emb_channels,
        dims=2,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=0,
        concat_channels = 0,
        guidance_drop_prob=0,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        scale_skip_connection=False,
        embedders=None,
        matryoshka_cuts=[1],
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        assert len(num_res_blocks) == len(channel_mult), f"num_res_blocks must be same length as channel_mult"

        self.input_size = input_size
        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.concat_channels = concat_channels
        self.guidance_drop_prob = guidance_drop_prob
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.scale_skip_connection = scale_skip_connection
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.matryoshka_cuts = matryoshka_cuts
            
        self.init_params = self.get_init_params()
            
        assert matryoshka_cuts[0] == 1, "first matryoshka cut must always be 1"
        self.next_matryoshka_cut = self.matryoshka_cuts[1] if len(self.matryoshka_cuts) > 1 else None
        if self.next_matryoshka_cut is not None:
            cut_index = int(math.log2(self.next_matryoshka_cut)+1)
            assert self.channel_mult[cut_index-1] == self.channel_mult[cut_index], "channel_mult must be the same before and after matryoshka cut"
            self.channel_mult = self.channel_mult[:cut_index]
            self.num_res_blocks = self.num_res_blocks[:cut_index]
        
        if embedders is None:
            self.time_embed = nn.Sequential(
                linear(self.model_channels, self.emb_channels),
                nn.SiLU(),
                linear(self.emb_channels, self.emb_channels),
            )

            if self.num_classes > 0:
                self.class_embed = nn.Sequential(
                    linear(self.num_classes, self.emb_channels),
                    nn.SiLU(),
                    linear(self.emb_channels, self.emb_channels),
                )
                self.null_cls_embed = torch.nn.Parameter(torch.randn(1, self.emb_channels))
        else:
            self.time_embed = embedders.get("time_embed")
            if self.num_classes > 0:
                self.class_embed = embedders.get("class_embed")
                self.null_cls_embed = embedders.get("null_cls_embed")

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(self.dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.emb_channels,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        scale_skip_connection=self.scale_skip_connection
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.emb_channels,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            scale_skip_connection=self.scale_skip_connection
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if self.next_matryoshka_cut is None:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    self.emb_channels,
                    self.dropout,
                    dims=self.dims,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    scale_skip_connection=self.scale_skip_connection
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=self.use_checkpoint,
                    num_heads=self.num_heads,
                    num_head_channels=self.num_head_channels,
                    use_new_attention_order=self.use_new_attention_order,
                ),
                ResBlock(
                    ch,
                    self.emb_channels,
                    self.dropout,
                    dims=self.dims,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    scale_skip_connection=self.scale_skip_connection
                ),
            )
        else:
            self.middle_block = self.create_inner_unet()
            pass

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.emb_channels,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        scale_skip_connection=self.scale_skip_connection
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.emb_channels,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            scale_skip_connection=self.scale_skip_connection
                        )
                        if self.resblock_updown
                        else Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def get_init_params(self):
        init_params = inspect.signature(self.__init__).parameters
        return {name: getattr(self, name) for name in init_params if hasattr(self, name)}
            
    def create_inner_unet(self):
        init_params = copy(self.init_params)
        next_matryoshka_cut = init_params['matryoshka_cuts'][1]
        current_matryoshka_blocks = len(self.channel_mult)
        init_params['matryoshka_cuts'] = [i//next_matryoshka_cut for i in init_params['matryoshka_cuts'][1:]]
        init_params['channel_mult'] = init_params['channel_mult'][current_matryoshka_blocks:]
        init_params['num_res_blocks'] = init_params['num_res_blocks'][current_matryoshka_blocks:]
        init_params['attention_resolutions'] = [i//2**current_matryoshka_blocks for i in init_params['attention_resolutions']]
        init_params['embedders'] = {
            "time_embed": self.time_embed,
            "class_embed": self.class_embed if self.num_classes > 0 else None,
            "null_cls_embed": self.null_cls_embed if self.num_classes > 0 else None,
        }
        init_params['input_size'] = self.input_size//2**current_matryoshka_blocks

        return UNetModel(**init_params)

    def forward(self, x, *args, **kwargs): # rounter for selecting the starting level only!
        if isinstance(x, torch.Tensor): # non-matryoshka run
            x = [x]
            if "concat" in kwargs:
                kwargs["concat"] = [kwargs["concat"]]
        
        if "concat" not in kwargs:
            kwargs["concat"] = [None] * len(x)
        
        total_matryoshka_levels = len(self.init_params['matryoshka_cuts'])
        if len(x) == total_matryoshka_levels:
            return self._forward(x, *args, **kwargs)[1]
        else:
            return self.middle_block(x, *args, **kwargs)

    def _forward(self, x, timesteps, cls=None, cls_embed=None, concat=None, drop_cls_prob=None, inner_matryoshka_run=False, matryoshka_emb=None, matryoshka_h=None, matryoshka_outputs=[]):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cls: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        device = self._get_model_device()
        dtype = self._get_model_dtype()
        batch_size = x[0].shape[0]

        if torch.is_grad_enabled(): # only check if we're in training mode
            assert (cls is not None) == (
                self.num_classes > 0
            ), "must specify cls if and only if the model is class-conditional"

        assert (concat[-1] is None and self.concat_channels == 0) or (concat[-1].shape[1] == self.concat_channels), "Number of concat channels do not match with initialized configuration."
        
        if not inner_matryoshka_run:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(dtype=dtype))

            if self.num_classes > 0:
                if cls is None and cls_embed is None: # usually the case for DDIM inversion
                    cls_embed = self.null_cls_embed.to(dtype)
                    drop_cls_prob = 0.0
                elif cls_embed is None:
                    assert cls.shape == (batch_size,self.num_classes)
                    cls_embed = self.class_embed(cls)

                drop_cls_prob = self.guidance_drop_prob if drop_cls_prob is None else drop_cls_prob

                cls_retention_mask = self._prob_mask_like(batch_size, 1-drop_cls_prob, device).unsqueeze(-1)
                cls_embed = torch.where(
                    cls_retention_mask,
                    cls_embed,
                    self.null_cls_embed.to(dtype)
                )

                emb = emb + cls_embed
        else:
            assert matryoshka_emb is not None, "matryoshka_emb must be specified in inner matryoshka run"
            emb = matryoshka_emb

        h = x[-1] if concat[-1] is None else torch.cat([x[-1],concat[-1]], dim=1)
        hs = []
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if i == 0 and matryoshka_h is not None:
                h += matryoshka_h
            hs.append(h)
        
        if len(x) > 1:
            h, matryoshka_outputs = self.middle_block._forward(x[:-1], timesteps, concat=concat[:-1], inner_matryoshka_run=True, matryoshka_emb=emb, matryoshka_h=hs[-1], matryoshka_outputs=matryoshka_outputs)
        else:
            h = self.middle_block(h, emb)
        
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        matryoshka_outputs.append(self.out(h))
        return h, matryoshka_outputs

    def forward_with_cond_scale(self, x, timesteps, cls=None, cls_embed=None, concat=None, cond_scale=0, phi=0.7):
        """model forward with conditional scale (used in inference)

        Args:
            x: input
            timesteps: timesteps
            cls: class vector. Defaults to None.
            cond_scale (int, optional): conditioning scale. Defaults to 0.
            phi (float, optional): rescale factor. Defaults to 0.7.

        Returns:
            _type_: model outputs after conditioning
        """
        device = self._get_model_device()
        dtype = self._get_model_dtype()
        batch_size = x[0].shape[0]

        if cond_scale == 0:
            return self._forward(x, timesteps, cls, cls_embed=cls_embed, concat=concat, drop_cls_prob=0)
        else:
            if cls is not None and cls_embed is None:
                cls_embed = self.class_embed(cls)
            elif cls_embed is not None:
                cls_embed = cls_embed
            else:
                cls_embed = self.null_cls_embed.to(dtype)
                
            # repeat null_cls_embed along batch dimension before concatenating
            null_cls_embed = self.null_cls_embed.repeat(batch_size, 1).to(dtype).to(device)
            cls_embed = torch.cat([cls_embed, null_cls_embed], dim=0)

            x = torch.cat([x, x], dim=0) # TODO: check if this is correct
            if concat is not None:
                concat = torch.cat([concat, concat], dim=0)
                
            timesteps = torch.cat([timesteps, timesteps], dim=0)
            out = self._forward(x, timesteps, cls=None, drop_cls_prob=0, cls_embed=cls_embed, concat=concat)
            logits, null_logits = torch.chunk(out, 2, dim=0)
            
            ccf_g = logits + (logits - null_logits) * cond_scale
            
            # Rescale classifier-free guidance (https://arxiv.org/abs/2305.08891)
            sigma_pos = torch.std(logits)
            sigma_ccf_g = torch.std(ccf_g)
            rescaled = ccf_g * (sigma_pos / sigma_ccf_g)
            final = phi * rescaled + (1 - phi) * ccf_g
            
            return final

    def _prob_mask_like(self, shape, prob, device):
        """generate a mask with prob
        
        Args:
            shape: shape of the mask
            prob: probability of the mask
            device: device of the mask

        Returns:
            mask: probablity mask (1: keep, 0: drop)
        """
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def _get_model_device(self):
        return next(self.parameters()).device
    
    def _get_model_dtype(self):
        return next(self.parameters()).dtype