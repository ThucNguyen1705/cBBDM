import itertools
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel

def disabled_train(self, mode=True):
    return self

class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        # SỬA ĐỔI: Hỗ trợ cross_attention cho RLI-DM
        elif self.condition_key == 'SpatialRescaler' or self.condition_key == 'cross_attention':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler' or self.condition_key == 'cross_attention':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond_sar, x_cond_herringbone, context=None):
        with torch.no_grad():
            # 1. Target (Optical) -> Latent 3 kênh
            x_latent = self.encode(x, cond=False)
            
            # 2. SAR -> Latent 3 kênh (Dữ liệu nguồn cho BBDM)
            # Đây là 'y' trong BBDM, không ghép với xương cá nữa
            x_cond_sar_latent = self.encode(x_cond_sar, cond=True)

        # 3. Xương cá -> Feature Map (Context cho Cross-Attention)
        # Đi qua mạng SpatialRescaler
        context_guidance = self.get_cond_stage_context(x_cond_herringbone)

        # Gọi forward lớp cha:
        # x=Target, y=SAR (3 kênh), context=Xương cá
        return super().forward(x_latent.detach(), x_cond_sar_latent.detach(), context_guidance)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond_sar, x_cond_herringbone, clip_denoised=False, sample_mid_step=False):
        # 1. Encode SAR (3 kênh)
        x_cond_sar_latent = self.encode(x_cond_sar, cond=True)

        # 2. Encode Xương cá (Context)
        context_guidance = self.get_cond_stage_context(x_cond_herringbone)
        
        # Tạo target shape (3 kênh)
        b, _, h, w = x_cond_sar_latent.shape
        target_shape = (b, 3, h, w)
        
        # Truyền y=SAR, context=Xương cá
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_sar_latent,
                                                     x_T_shape=target_shape,
                                                     context=context_guidance,
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True, smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))
            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps", dynamic_ncols=True, smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_sar_latent,
                                      x_T_shape=target_shape,
                                      context=context_guidance,
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec