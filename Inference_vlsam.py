# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer,BertModel,AutoProcessor,MambaModel,BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, AutoModel
from utils_downstream.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()


def eval_psnr(loader, model,vlm_model,processor,mamba_model,tokenizer,eval_type=None,device=None,save_path=None):
    model.eval()

    captions = {}   # stem -> caption string, saved to captions.json

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
      
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    
    mae,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()

    for step, (image, gt2D, img_1024_ori, img_path, has_gt) in enumerate(loader):

        image, gt2D = image.to(device), gt2D.to(device)
        img_1024_ori = img_1024_ori.to(device)
        with torch.no_grad():
        
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs,output_hidden_states=True)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)
            
            ### Extract text information
            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            
            mamba_outputs = mamba_model(**mamba_inputs)
            
            text_features = mamba_outputs.last_hidden_state
            
            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features = vision_outputs.last_hidden_state[:,1:,:]
            
            pred = torch.sigmoid(model(image,text_features,image_features))

            # always record caption (keyed by stem relative to frames_dir)
            rel = os.path.relpath(img_path[0], start=loader.dataset.frames_dir)
            stem_key = os.path.splitext(rel)[0]
            captions[stem_key] = description

            if save_path is not None:
                mask_np = (pred.squeeze().cpu().numpy() > 0.5).astype('uint8') * 255
                out_path = os.path.join(save_path, stem_key + '.png')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(mask_np).save(out_path)

            if has_gt.item():
                res = pred.squeeze().squeeze().cpu().numpy()
                gt = gt2D.squeeze().squeeze().cpu().numpy()
                mae.update(res, gt)
                sm.update(res, gt)
                em.update(res, gt)
                wfm.update(res, gt)
                m_dice.update(res, gt)
                m_iou.update(res, gt)
                ber.update(res, gt)
        
        if pbar is not None:
            pbar.update(1)

    MAE = mae.show()    
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    ber = ber.show()

    if pbar is not None:
        pbar.close()

    # Save captions alongside the predictions
    if save_path is not None and captions:
        captions_path = os.path.join(save_path, "captions.json")
        with open(captions_path, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"[info] Captions saved to {captions_path}")

    return sm, em, wfm, MAE
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, bbox_shift=20):
        # Collect all frames recursively
        all_frames = []
        for root, _, files in os.walk(frames_dir):
            for f in sorted(files):
                if f.endswith('.jpg') or f.endswith('.png'):
                    all_frames.append(join(root, f))
        all_frames.sort()

        # Build index of available GT masks: relative path (no ext) -> mask path
        mask_index = {}
        for root, _, files in os.walk(masks_dir):
            for f in files:
                if f.endswith('.png'):
                    rel_subdir = os.path.relpath(root, masks_dir)
                    stem = os.path.splitext(f)[0]
                    # When GT is a flat folder, rel_subdir == '.' — normalise
                    # so the key matches the frame lookup key (stem only).
                    if rel_subdir == '.':
                        key = stem
                    else:
                        key = join(rel_subdir, stem)
                    mask_index[key] = join(root, f)

        self.img_path_files = all_frames
        self.gt_path_files  = []
        for fp in all_frames:
            rel = os.path.relpath(fp, frames_dir)
            key = os.path.splitext(rel)[0]
            self.gt_path_files.append(mask_index.get(key, None))  # None if no GT

        self.frames_dir = frames_dir
        self.bbox_shift = bbox_shift
        n_with_gt = sum(1 for g in self.gt_path_files if g is not None)
        print(f"total frames: {len(all_frames)}, with GT mask: {n_with_gt}")
        
        self.img_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):

        img_1024_ori = Image.open(self.img_path_files[index]).convert('RGB')
        img_1024 = self.img_transform(img_1024_ori)

        gt_path = self.gt_path_files[index]
        if gt_path is not None:
            gt = self.mask_transform(Image.open(gt_path).convert('L'))
            has_gt = torch.tensor(1)
        else:
            gt = torch.zeros(1, 1024, 1024)
            has_gt = torch.tensor(0)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            np.array(img_1024_ori),
            self.img_path_files[index],
            has_gt,
        )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--frames_path",
    type=str,
    default="/Experiments/marcol01/frames",
    help="path to directory containing input frames (.jpg/.png)",
)
parser.add_argument(
    "--masks_path",
    type=str,
    default="/Experiments/marcol01/masks",
    help="path to directory containing GT masks (.png)",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="path to directory where predicted masks will be saved (optional)",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument(
    "-checkpoint", type=str, default="./work_dir/SOTA_real/vlsam_model_best.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")

parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


        
class VLSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.pseudo_mask_embed =  nn.Sequential(
                nn.Conv2d(256, 256,3,1,1),
                nn.GELU())


    def forward(self, image,text_embeddings,image_features):
        blip_img_adap = image_features.reshape(1,-1,64,64)
        image_embedding = self.image_encoder(image, blip_img_adap)  # (B, 256, 64, 64)

        mamba_text = text_embeddings.reshape(1,-1,256)
        blip_img = image_features.reshape(1,-1,256)
        sparse_embeddings = torch.cat((mamba_text,blip_img),dim=1)
       
   
        bs,c,h,w = image_embedding.shape
        dense_embeddings = self.pseudo_mask_embed(image_embedding)
       
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.pe_layer((64,64)).unsqueeze(0),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():

    sam_model = sam_model_registry[args.model_type](checkpoint='work_dir/SAM/sam_vit_h_4b8939.pth')
    vlsam_model = VLSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    vlsam_model.load_state_dict(checkpoint["model"])
    
    test_dataset = NpyDataset(args.frames_path, args.masks_path)
    print("Number of test samples: ", len(test_dataset))

    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        #num_workers=args.num_workers,
        pin_memory=True,
    )

            
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    mamba_model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf").to(device)
    
            
    result1, result2, result3, result4 = eval_psnr(test_dataloader, vlsam_model,vlm_model,processor,mamba_model,tokenizer,
                eval_type='cod',device=device,save_path=args.save_path)
    print({'Sm': result1})
    print({'Em': result2})
    print({'wFm': result3})
    print({'Mae': result4})




if __name__ == "__main__":
    main()
