import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import cv2
import torchvision.transforms as T
import imagesize


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="config file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, '*.png')))
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("logs/2023-01-03T13-47-43_autoencoder_kl_64x64x3_custom/checkpoints/epoch=000056.ckpt")["state_dict"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.eval()
    model = model.to(device)
    # sampler = DDIMSampler(model)
    os.makedirs(opt.outdir, exist_ok=True)
    transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)])
    with torch.no_grad():
        for mask in tqdm(masks):
            outpath = os.path.join(opt.outdir, os.path.split(mask)[1])
            original_size = imagesize.get(mask)
            image = transform(Image.open(mask).convert("L"))[None].to(device)
            z = model(image)
            inpainted = ((z[0][0,0].clamp(-1, 1) + 1)/2 * 255.).cpu().numpy()
            resized = cv2.resize(inpainted.astype(np.uint8), dsize=original_size, interpolation=cv2.INTER_LINEAR)
            Image.fromarray(resized).save(outpath)