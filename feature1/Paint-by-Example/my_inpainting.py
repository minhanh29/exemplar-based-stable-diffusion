import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize
wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.cpu()
    model.eval()
    # torch.save({
    #     "state_dict": model.state_dict()
    # }, "./checkpoints/my_sd_model.ckpt")
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class ImageInpainting:
    def __init__(self,
                 ckpt,
                 plms=True,
                 config_path="./Paint-by-Example/configs/v1.yaml"):
        seed_everything(321)

        config = OmegaConf.load(f"{config_path}")
        model = load_model_from_config(config, f"{ckpt}")

        self.model = model.to(device)

        if plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

    def predict(self, image_path, mask_path, reference_path, result_path,
                n_samples=1, precision="autocast", scale=5, ddim_steps=50):
        C = 4
        f = 8
        H = 512
        W = 512
        start_code = None
        precision_scope = autocast if precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    img_p = Image.open(image_path).convert("RGB")
                    img_p = expand2square(img_p, (0, 0, 0)).resize((512, 512))
                    image_tensor = get_tensor()(img_p)
                    image_tensor = image_tensor.unsqueeze(0)

                    ref_p = Image.open(reference_path).convert("RGB").resize((224,224))
                    ref_tensor=get_tensor_clip()(ref_p)
                    ref_tensor = ref_tensor.unsqueeze(0)

                    mask=Image.open(mask_path)
                    mask = expand2square(mask, (0, 0, 0)).resize((512, 512))
                    mask = mask.convert("L")
                    mask = np.array(mask)[None,None]
                    mask = 1 - mask.astype(np.float32)/255.0
                    mask[mask < 0.5] = 0
                    mask[mask >= 0.5] = 1
                    mask_tensor = torch.from_numpy(mask)
                    inpaint_image = image_tensor*mask_tensor
                    test_model_kwargs={}
                    test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
                    test_model_kwargs['inpaint_image']=inpaint_image.to(device)
                    ref_tensor=ref_tensor.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = self.model.learnable_vector
                    c = self.model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = self.model.proj_out(c)
                    inpaint_mask = test_model_kwargs['inpaint_mask']
                    z_inpaint = self.model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = self.model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [C, H // f, W // f]
                    samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=0.0,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
                    def un_norm_clip(x):
                        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
                        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
                        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
                        return x

                    for i,x_sample in enumerate(x_checked_image_torch):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(result_path)
                    del img_p, mask, ref_p

        print(f"Your samples are ready and waiting for you here: \n{result_path} \n")

