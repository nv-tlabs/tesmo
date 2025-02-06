from diffusion.respace import SpacedDiffusion
from .gaussian_diffusion import _extract_into_tensor, GaussianDiffusion
import torch as th

class InpaintingGaussianDiffusion(SpacedDiffusion):
    def q_sample(self, x_start, t, noise=None, model_kwargs=None): # ! is used for initialization
        # import pdb;pdb.set_trace()
        """
        overrides q_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        bs, feat, _, frames = noise.shape
        noise *= 1. - model_kwargs['y']['inpainting_mask']
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        # import pdb;pdb.set_trace()
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        noise *= 1. - model_kwargs['y']['inpainting_mask']

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        
        
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        import pdb;pdb.set_trace()
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

### training extra losses;
class InpaintingGaussianDiffusionOnlyGoal(SpacedDiffusion):
# class InpaintingGaussianDiffusionOnlyGoal(GaussianDiffusion):
    def __init__(self, use_timesteps, indicator=False, start_end_pose=True, **kwargs):
        
        self.use_indicator = indicator
        self.start_end_pose = start_end_pose
        self.mask_out = kwargs.get('mask_out', True)
        
        
        self.startEnd_addNoise = kwargs.get('cfg_startEnd_addNoise', False)
        # self.cond_mask_prob_startEnd = kwargs.get('cfg_startEnd_addNoise', False)
        #     import pdb;pdb.set_trace()
        #     startEnd_mask = th.bernoulli(torch.ones(bs, device=y['mask'].device) * self.cond_mask_prob_startEnd).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
        #     y['startEnd_addNoise'] = startEnd_mask.bool()
        
        print('inpainting mask out: ', self.mask_out)
        super().__init__(use_timesteps, **kwargs)
        print(f'cfg on addNoise on the start and end: {self.startEnd_addNoise}; scale: {self.cond_mask_prob_startEnd}')
        
    def q_sample(self, x_start, t, noise=None, model_kwargs=None, infer_startEnd_addNoise=False): # sampling initialization; and every training steps.
        """
        overrides q_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        
        # import pdb;pdb.set_trace()
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        bs, feat, _, frames = noise.shape
        
        # get the inference motion condition mask.
        import pdb;pdb.set_trace()
        cfg_motion_control_addNoise=model_kwargs['y'].get('uncond_motion_control_addNoise', False)    
        
        if self.mask_out:
            # import pdb;pdb.set_trace()
            tmp_mask = infer_startEnd_addNoise
            if self.startEnd_addNoise:
                startEnd_mask = th.bernoulli(th.ones(bs, device=tmp_mask.device) * self.cond_mask_prob_startEnd).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
                tmp_mask = tmp_mask * (1-startEnd_mask).reshape(bs, 1, 1, 1)
            
            if cfg_motion_control_addNoise:
                tmp_mask = tmp_mask * 0.0 # set mask as zero.
                
            for tmp_i in range(model_kwargs['y']['lengths'].shape[0]):
                tmp_l = model_kwargs['y']['lengths'][tmp_i]  
                tmp_mask[tmp_i, :, :, 1:tmp_l-1] *= 0.0 # only the last frame.
                
                # import pdb;pdb.set_trace()
                if self.use_indicator: # does not model indicator information.
                    tmp_mask[tmp_i, -1, :, 1:tmp_l-1] = 1.0
                    
        else: # no mask at all; without predicting indicator.
            tmp_mask = model_kwargs['y']['inpainting_mask'] * 0.0
            for tmp_i in range(model_kwargs['y']['lengths'].shape[0]):
                tmp_l = model_kwargs['y']['lengths'][tmp_i]
                if self.use_indicator: # does not model indicator information.
                    tmp_mask[tmp_i, -1, :, 1:tmp_l-1] = 1.0

        # TODO: add random dropouts for predicting noise on the input start frames and end frames.
        
        
        noise *= 1. - tmp_mask
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        infer_startEnd_addNoise=False
    ):
        # import pdb;pdb.set_trace()
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        
        # get the inference motion condition mask.
        import pdb;pdb.set_trace()
        # cfg_motion_control_addNoise=model_kwargs['y'].get('uncond_motion_control_addNoise', False)    
        cfg_motion_control_addNoise = infer_startEnd_addNoise
        
        if self.mask_out:
            tmp_mask = model_kwargs['y']['inpainting_mask']
            if cfg_motion_control_addNoise:
                tmp_mask = tmp_mask * 0.0 # set mask as zero.
                
            for tmp_i in range(model_kwargs['y']['lengths'].shape[0]):
                tmp_l = model_kwargs['y']['lengths'][tmp_i] # all information.
                tmp_mask[tmp_i, :, :, 1:tmp_l-1] *= 0.0 # leave the last one.
                if self.use_indicator:
                    tmp_mask[tmp_i, -1, :, 1:tmp_l-1] = 1.0
        else:
            tmp_mask = model_kwargs['y']['inpainting_mask'] * 0.0
            for tmp_i in range(model_kwargs['y']['lengths'].shape[0]):
                tmp_l = model_kwargs['y']['lengths'][tmp_i]
                if self.use_indicator: # does not model indicator information.
                    tmp_mask[tmp_i, -1, :, 1:tmp_l-1] = 1.0
            
                    
        noise *= 1. - tmp_mask

        # import pdb;pdb.set_trace()
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        
        # import pdb;pdb.set_trace()
        # TODO: fix bugs in classifer guidance.
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        
        # import pdb;pdb.set_trace()
        sample = out["mean"] + nonzero_mask * th.expsample(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_cfg_twice(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        infer_startEnd_addNoise=False
    ):
        # import pdb;pdb.set_trace()
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        
        # get the inference motion condition mask.
        import pdb;pdb.set_trace()
        # cfg_motion_control_addNoise=model_kwargs['y'].get('uncond_motion_control_addNoise', False)    
        cfg_motion_control_addNoise = infer_startEnd_addNoise
        
        if self.mask_out:
            tmp_mask = model_kwargs['y']['inpainting_mask']
            if cfg_motion_control_addNoise:
                tmp_mask = tmp_mask * 0.0 # set mask as zero.
                
            for tmp_i in range(model_kwargs['y']['lengths'].shape[0]):
                tmp_l = model_kwargs['y']['lengths'][tmp_i] # all information.
                tmp_mask[tmp_i, :, :, 1:tmp_l-1] *= 0.0 # leave the last one.
                if self.use_indicator:
                    tmp_mask[tmp_i, -1, :, 1:tmp_l-1] = 1.0
        else:
            tmp_mask = model_kwargs['y']['inpainting_mask'] * 0.0
            for tmp_i in range(model_kwargs['y']['lengths'].shape[0]):
                tmp_l = model_kwargs['y']['lengths'][tmp_i]
                if self.use_indicator: # does not model indicator information.
                    tmp_mask[tmp_i, -1, :, 1:tmp_l-1] = 1.0
            
                    
        noise *= 1. - tmp_mask

        # import pdb;pdb.set_trace()
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        
        # import pdb;pdb.set_trace()
        # TODO: fix bugs in classifer guidance.
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        
        # import pdb;pdb.set_trace()
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

### inference guidance;    
class InpaintingGaussianDiffusionGudiance(SpacedDiffusion):
    def q_sample(self, x_start, t, noise=None, model_kwargs=None): # ! is used for initialization
        # import pdb;pdb.set_trace()
        """
        overrides q_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        bs, feat, _, frames = noise.shape
        noise *= 1. - model_kwargs['y']['inpainting_mask']
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        # import pdb;pdb.set_trace()
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        noise *= 1. - model_kwargs['y']['inpainting_mask']

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        # TODO: add guidance inference.
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        
        # import pdb;pdb.set_trace()
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

