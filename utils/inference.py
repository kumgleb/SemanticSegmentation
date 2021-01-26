import toml
import torch
import numpy as np
from PIL import Image
from collections import namedtuple
from skimage.transform import resize
from torch.utils import model_zoo
from dataloader.transformations import Normalize
from models.model_unet import UNet
from models.model_resnet_unet import ResUNet


class TrainedModel:
    """
    Class that load pre-trained model and perform mask inference.
    Available models:
        UNet: plain UNet.
        ResUNet: UNet with ResNet34 as encoder.
    """

    model_tpl = namedtuple('model', ['url', 'model'])
    models = {
        'UNet': model_tpl(
            url='https://github.com/kumgleb/SemanticSegmentation/releases/download/%23trained_unet/UNet_0.295.pth',
            model=UNet
        ),
        'ResUNet': model_tpl(
            url='https://github.com/kumgleb/SemanticSegmentation/releases/download/%23trained_resunet/ResUNet_0.192.pth',
            model=ResUNet
        )
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.get_model(model_name)
        self.input_img_size = (256, 256) if model_name == 'UNet' else (224, 224)
        self.original_img_size = None

    def get_model(self, model_name: str):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cfg = toml.load('./configs/cfg.toml')[model_name]
        model = self.models[model_name].model(cfg).to(device)
        state_dict = model_zoo.load_url(self.models[model_name].url, progress=True, map_location=device)
        model.load_state_dict(state_dict)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(device)
        model.eval()
        return model

    def preprocess_img(self, img):
        img = np.asanyarray(img, dtype='uint8')
        self.original_img_size = img.shape[:2]
        img, _ = Normalize()(img, [])
        img = resize(img, self.input_img_size)
        img = np.moveaxis(img, -1, 0)
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img).type(torch.float32)
        return img

    def predict_mask(self, img):
        self.model.eval()
        with torch.no_grad():
            mask_prd = self.model(img)
        mask_prd = mask_prd.argmax(dim=1)
        mask_prd = mask_prd.cpu().numpy()
        return mask_prd

    def eval_mask(self, img: str):
        img = Image.open(img)
        img = self.preprocess_img(img)
        mask = self.predict_mask(img)
        mask = resize(mask[0], self.original_img_size)
        return mask
