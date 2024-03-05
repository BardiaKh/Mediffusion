import torch
import torch.nn as nn

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
    :param missing_class_value: if specified, then this value will be replaced by
        a learned parameter. This is useful for missing classes in the dataset.
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
        dims=2,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=0,
        concat_channels = 0,
        missing_class_value = None,
        guidance_drop_prob=0,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        scale_skip_connection=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        assert len(num_res_blocks) == len(channel_mult), f"num_res_blocks must be same length as channel_mult"

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
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
        self.missing_class_value = missing_class_value
        
        if missing_class_value is not None:
            self.missing_class_param = torch.nn.Parameter(torch.randn(1))

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes > 0:
            class_embed_dim = time_embed_dim
            self.class_embed = nn.Sequential(
                linear(self.num_classes, class_embed_dim),
                nn.SiLU(),
                linear(class_embed_dim, class_embed_dim),
            )
            self.null_cls_embed = torch.nn.Parameter(torch.randn(1, class_embed_dim))

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        scale_skip_connection=scale_skip_connection
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            scale_skip_connection=scale_skip_connection
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                scale_skip_connection=scale_skip_connection
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                scale_skip_connection=scale_skip_connection
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        scale_skip_connection=scale_skip_connection
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            scale_skip_connection=scale_skip_connection
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, cls=None, cls_embed=None, concat=None, drop_cls_prob=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cls: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if torch.is_grad_enabled(): # only check if we're in training mode
            assert (cls is not None) == (
                self.num_classes > 0
            ), "must specify cls if and only if the model is class-conditional"

        assert (concat is None and self.concat_channels == 0) or (concat.shape[1] == self.concat_channels), "Number of concat channels do not match with initialized configuration."

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(dtype=x.dtype))

        if self.num_classes > 0:
            if cls is None and cls_embed is None: # usually the case for DDIM inversion
                cls_embed = self.null_cls_embed.to(x.dtype)
                drop_cls_prob = 0.0
            elif cls_embed is None:
                assert cls.shape == (x.shape[0],self.num_classes)

                if self.missing_class_value is not None:
                    cls = torch.where(
                        cls == self.missing_class_value, 
                        self.missing_class_param, 
                        cls
                    )

                cls_embed = self.class_embed(cls)

            drop_cls_prob = self.guidance_drop_prob if drop_cls_prob is None else drop_cls_prob

            cls_retention_mask = self._prob_mask_like(x.shape[0], 1-drop_cls_prob, x.device).unsqueeze(-1)
            cls_embed = torch.where(
                cls_retention_mask,
                cls_embed,
                self.null_cls_embed.to(x.dtype)
            )

            emb = emb + cls_embed

        h = x if concat is None else torch.cat([x,concat], dim=1)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        h = self.out(h)
        
        return h

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
        if cond_scale == 0:
            return self(x, timesteps, cls, cls_embed=cls_embed, concat=concat, drop_cls_prob=0)
        else:
            if cls is not None and cls_embed is None:
                cls_embed = self.class_embed(cls)
            elif cls_embed is not None:
                cls_embed = cls_embed
            else:
                cls_embed = self.null_cls_embed.to(x.dtype)
                
            # repeat null_cls_embed along batch dimension before concatenating
            null_cls_embed = self.null_cls_embed.repeat(x.shape[0], 1).to(x.dtype)
            cls_embed = torch.cat([cls_embed, null_cls_embed], dim=0)

            x = torch.cat([x, x], dim=0)
            if concat is not None:
                concat = torch.cat([concat, concat], dim=0)
                
            timesteps = torch.cat([timesteps, timesteps], dim=0)
            out = self(x, timesteps, cls=None, drop_cls_prob=0, cls_embed=cls_embed, concat=concat)
            logits, null_logits = torch.chunk(out, 2, dim=0)
            
            ccf_g = null_logits + (logits - null_logits) * cond_scale
            
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
