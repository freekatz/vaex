import PIL.Image as PImage
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import InterpolationMode, transforms

from utils.data_sampler import DistInfiniteBatchSampler
from utils import dist_utils
from utils import arg_util
from utils.my_dataset import UnlabeledDatasetFolder, FFHQ

PImage.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def load_pil(path: str, proposal_size):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        w: int = img.width
        h: int = img.height
        sh: int = min(h, w)
        if sh > proposal_size:
            ratio: float = proposal_size / sh
            w = round(ratio * w)
            h = round(ratio * h)
        img.draft('RGB', (w, h))
        img = img.convert('RGB')
    return img


def build_transforms(args: arg_util.Args):
    final_reso=args.data_load_reso
    mid_reso=args.mid_reso
    hflip=args.hflip
    dataset_name = args.dataset_name
    if dataset_name in ['imagenet', 'ffhq']:
        # build augmentations
        mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
        train_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
        if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())

        val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
    else:
        raise NotImplementedError
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    return train_aug, val_aug


def build_dataset(
    dataset_name: str, data_path: str, transform: transforms.Compose, split='train'
):
    if dataset_name == 'imagenet':
        dataset = UnlabeledDatasetFolder(root=data_path, transform=transform, split=split)
    elif dataset_name == 'ffhq':
        dataset = FFHQ(root=data_path, transform=transform, split=split)
    else:
        raise NotImplementedError
    print(f'[Dataset] {len(dataset)=}')
    return dataset


def build_data_loader(args, start_ep, start_it, dataset=None, transform=None, split='train'):
    if dataset is None:
        dataset = build_dataset(args.dataset_name, args.data_path, transform, split=split)
    if split == 'train':
        data_loader = DataLoader(
            dataset=dataset, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(),  # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset), glb_batch_size=args.bs,
                same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist_utils.get_rank(), world_size=dist_utils.get_world_size(),
                start_ep=start_ep, start_it=start_it,
            ),
        )
    else:
        raise NotImplementedError
    return data_loader


def no_transform(x): return x


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
