from torch.utils.data import DataLoader

from data_loaders.amass.sampling import FrameSampler
from data_loaders.tensors import collate as all_collate, babel_collate, babel_eval_collate, pw3d_collate
from data_loaders.tensors import t2m_collate, t2m_collate_eval
from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.data.dataset import collate_fn as sorted_collate
from data_loaders.dataloader import build_dataloader

SAVE_THINGS_FOR_SPERATE_EVALUATOR = True

def get_dataset_class(name, load_mode):
    if name == "babel":
        if load_mode == "text_only":
            load_mode = 'train'
        if load_mode in ['gt', 'eval', 'movement_train', 'evaluator_train']:
            from data_loaders.humanml.data.dataset import BABEL_eval
            return BABEL_eval
        elif load_mode == 'train':
            from data_loaders.amass.babel import BABEL
            return BABEL
        else:
            raise ValueError(f'Unsupported load_moad name [{load_mode}]')
    elif name == "humanml":
        from data_loaders.humanml.data.dataset_wrapper import HumanML3D
        return HumanML3D
    elif name == 'humanmlabsolute': # we use the root translation + rot6d representation.
        from data_loaders.humanmlabsolute.data.dataset import HumanML3DAbsolute
        return HumanML3DAbsolute
    elif name == "pw3d":
        from data_loaders.humanml.data.dataset import PW3D
        return PW3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, load_mode='train'):
    if load_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit", "humanmlabsolute"]:
        if SAVE_THINGS_FOR_SPERATE_EVALUATOR:
            return t2m_collate_eval
        else:
            return t2m_collate
    elif name == 'pw3d':
        return pw3d_collate
    elif name == 'babel':
        if load_mode =='eval':
            return babel_eval_collate
        elif load_mode == 'movement_train':
            return default_collate
        elif load_mode == 'evaluator_train':
            return sorted_collate
        else:
            return babel_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', load_mode='train', batch_size=None, opt=None, short_db=False, cropping_sampler=False, size=None, opt_name=None):
    DATA = get_dataset_class(name, load_mode)

    # import pdb;pdb.set_trace()
    if name in ["humanml", "pw3d", "humanmlabsolute"]:
        dataset = DATA(split=split, num_frames=num_frames, load_mode=load_mode, size=size, datapath=opt_name, batch_size=batch_size)
    elif name == "babel": # not using this one.
        from data_loaders.amass.transforms import SlimSMPLTransform
        if ((split=='val') and (cropping_sampler==True)):
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh',
                                          normalization=True, canonicalize=False)
        else:
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        sampler = FrameSampler(min_len=num_frames[0], max_len=num_frames[1])
        dataset = DATA(split=split,
                       datapath='./dataset/babel/babel-smplh-30fps-male',
                       transforms=transform, load_mode=load_mode, mode='train', opt=opt, sampler=sampler,
                       short_db=short_db, cropping_sampler=cropping_sampler)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset

def get_dataset_loader(name, batch_size, num_frames, split='train', load_mode='train', opt=None, short_db=False, cropping_sampler=False, size=None, opt_name=None, is_distributed=False, gpus=1):
    if load_mode == 'text_only':
        load_mode = 'train'
    # import pdb;pdb.set_trace()
    dataset = get_dataset(name, num_frames, split, load_mode, batch_size, opt, short_db, cropping_sampler, size, opt_name=opt_name)
    collate = get_collate_fn(name, load_mode)

    print(collate)
    
    n_workers = 1 if load_mode in ['movement_train', 'evaluator_train'] else 8
    # import pdb;pdb.set_trace()
    
    if is_distributed:
        loader = build_dataloader(dataset,
                    samples_per_gpu=batch_size,
                    workers_per_gpu=1,
                    dist = True,
                    shuffle = True,
                    num_gpus=gpus,
                    drop_last=True,
                    collate_fn=collate,
                    )

    print(f'dataset load mode: {load_mode}')
    
    if split != 'train':
        print(f'split : {split}, without shuffle !!!!!!!!!!')
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=n_workers, drop_last=True, collate_fn=collate
        )
    else:
        loader = DataLoader(
            # dataset, batch_size=batch_size, shuffle=False,
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=n_workers, drop_last=True, collate_fn=collate
        )

    return loader