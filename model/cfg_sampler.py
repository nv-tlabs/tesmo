import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):
    """this is a wrapper around a model that applies classifier-free guidance when sampling.
    Note that when accessing the model's attributes, you must it returns the wrapped model's attributes.
    This does not apply to functions, though"""
    def __init__(self, model):
        super().__init__()
        vars(self)['model'] = model
        assert model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def __getattr__(self, name: str):
        model = vars(self)['model']
        return getattr(model, name)

    def forward(self, x, timesteps, y=None):
        # import pdb;pdb.set_trace()
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
    
    def parameters(self):
        return self.model.parameters()

class ClassifierFreeSampleModelMotionControl(nn.Module): 
    # on 2D floor maps | interactive object.

    """this is a wrapper around a model that applies classifier-free guidance when sampling.
    Note that when accessing the model's attributes, you must it returns the wrapped model's attributes.
    This does not apply to functions, though"""
    def __init__(self, model, motion_control_guidance_param=1.0):
        # import pdb;pdb.set_trace()
        super().__init__()
        vars(self)['model'] = model
        
        assert model.cond_mask_prob_noObj > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions' 
        self.motion_control_guidance_param = motion_control_guidance_param
        
    def __getattr__(self, name: str):
        model = vars(self)['model']
        return getattr(model, name)

    def forward(self, x, timesteps, y=None):
        # TODO: four types; consider CFG guidance on the text description.
        # for ! text information: we always conditioned one.

        y['uncond'] = False # with object 
        y_uncond = deepcopy(y)
        y_uncond['uncond_noObj'] = True # no condition: 10 % drop out
    
        # import pdb;pdb.set_trace()

        if self.motion_control_guidance_param == 1.0:
            if y['scale'][0] == 0.0:
                y['uncond'] = True
                out = self.model(x, timesteps, y) 
            elif y['scale'][0] == 1.0:
                y['uncond'] = False
                out = self.model(x, timesteps, y)
            else:
                y_text_uncond = deepcopy(y)
                y_text_uncond['uncond'] = True
                
                y['uncond'] = False
                out = self.model(x, timesteps, y)
                out_text_uncond = self.model(x, timesteps, y_text_uncond)
                out = out_text_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_text_uncond))
            return out
        
        elif self.motion_control_guidance_param == 0.0:
            if y['scale'][0] == 0.0:
                y_uncond['uncond'] = True
                out = self.model(x, timesteps, y_uncond) 
            elif y['scale'][0] == 1.0:
                y_uncond['uncond'] = False
                out = self.model(x, timesteps, y_uncond)
            else:
                y_uncond_text_uncond = deepcopy(y_uncond)
                y_uncond_text_uncond['uncond'] = True
                
                y_uncond['uncond'] = False
                out = self.model(x, timesteps, y_uncond)
                out_text_uncond = self.model(x, timesteps, y_uncond_text_uncond)
                out = out_text_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_text_uncond))
            return out

        else:
            # !!! this is the correct one.
            y_text_uncond = deepcopy(y)
            y_text_uncond['uncond'] = True  
            out_text_uncond = self.model(x, timesteps, y_text_uncond) # no text, with obj

            # out_text = self.model(x, timesteps, y) # with text, with obj
            # out = out_text_uncond + (y['scale'].view(-1, 1, 1, 1) * (out_text - out_text_uncond))    

            out_uncond_text = self.model(x, timesteps, y_uncond) # with text, no obj

            y_uncond_text_uncond = deepcopy(y_uncond) 
            y_uncond_text_uncond['uncond'] = True 
            out_uncond_text_uncond = self.model(x, timesteps, y_uncond_text_uncond) # no text, no obj

            return out_uncond_text_uncond + (y['scale'].view(-1, 1, 1, 1) * (out_uncond_text - out_uncond_text_uncond)) + (self.motion_control_guidance_param * (out_text_uncond - out_uncond_text_uncond))
            
                
        
    def parameters(self):
        return self.model.parameters()

class UnconditionedModel(nn.Module):
    """this is a wrapper around a model that forces unconditional sampling.
    Note that when accessing the model's attributes, you must it returns the wrapped model's attributes.
    This does not apply to functions, though"""
    def __init__(self, model):
        super().__init__()
        vars(self)['model'] = model
        assert model.cond_mask_prob > 0, 'Cannot run unconditional generation on a model that has not been trained with no conditions'

    def __getattr__(self, name: str):
        model = vars(self)['model']
        return getattr(model, name)

    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond
    
    def parameters(self):
        return self.model.parameters()

def wrap_model(model, args):
    # TODO: use CFG on the input start and end pose; 2D floor maps;
    # * add CFG on the object.
    # import pdb; pdb.set_trace()
    if args.cfg_motion_control:
        print('Classifier-free guidance is supported for motion control.')
        return ClassifierFreeSampleModelMotionControl(model, motion_control_guidance_param=args.motion_control_guidance_param)
    if args.guidance_param not in [0., 1.]:
        return ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    elif args.guidance_param == 0:
        return UnconditionedModel(model)
    else:
        return model
    