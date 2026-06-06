import os
import random
import glob

import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import crop, rotate
from torchvision.transforms import ToTensor, RandomCrop

random.seed(42)

def create_train_val_split(hazy_path, clear_path, val_ratio=0.1):
    hazy_files = glob.glob(os.path.join(hazy_path, "*.jpg"))
    scene_dict = {}
    for hazy_file in hazy_files:
        hazy_name = os.path.basename(hazy_file)
        base_name = os.path.splitext(hazy_name)[0]
        clear_name = '_'.join(base_name.split('_')[:-2]) + '.jpg'

        if clear_name not in scene_dict:
            scene_dict[clear_name] = []
        scene_dict[clear_name].append(hazy_name)

    scene_keys = list(scene_dict.keys())
    random.shuffle(scene_keys)
    split_idx = int(len(scene_keys) * (1 - val_ratio))
    train_keys = scene_keys[:split_idx]
    val_keys = scene_keys[split_idx:]
    train_list = []
    val_list = []

    for key in train_keys:
        for hazy_name in scene_dict[key]:
            train_list.append(
            (os.path.join(hazy_path, hazy_name), os.path.join(clear_path, key))
            )

    for key in val_keys:
        for hazy_name in scene_dict[key]:
            val_list.append(
                (os.path.join(hazy_path, hazy_name), os.path.join(clear_path, key))
            )
    random.shuffle(train_list)
    random.shuffle(val_list)

    print(f"Training scenes: {len(train_keys)}")
    print(f"Validation scenes: {len(val_keys)}")
    print(f"Training pairs: {len(train_list)}")
    print(f"Validation pairs: {len(val_list)}")
    return train_list, val_list


class TrainDataset(data.Dataset):
    def __init__(self, train_list):
        super(TrainDataset, self).__init__()
        self.train_list = train_list

    def __getitem__(self, index):
        hazy_path, clear_path = self.train_list[index]
        hazy = Image.open(hazy_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')
        # DEA-Net augmentation
        crop_params = RandomCrop.get_params(hazy, output_size=(256, 256))
        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)
        rotate_angle = random.randint(0, 3) * 90
        hazy = rotate(hazy, rotate_angle)
        clear = rotate(clear, rotate_angle)
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)
        return hazy, clear

    def __len__(self):
        return len(self.train_list)


class ValDataset(data.Dataset):
    def __init__(self, val_list):
        super(ValDataset, self).__init__()
        self.val_list = val_list

    def __getitem__(self, index):

        hazy_path, clear_path = self.val_list[index]
        hazy = Image.open(hazy_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')
        filename = os.path.basename(hazy_path)
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': filename}

    def __len__(self):
        return len(self.val_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.hazy_image_list.sort()

    def __getitem__(self, index):
        hazy_name = self.hazy_image_list[index]
        hazy_path = os.path.join(self.hazy_path, hazy_name)
        hazy = Image.open(hazy_path).convert('RGB')
        hazy = ToTensor()(hazy)
        return hazy, hazy_name

    def __len__(self):
        return len(self.hazy_image_list)
