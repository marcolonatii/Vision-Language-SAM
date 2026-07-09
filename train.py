# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
"""

# %% setup environment
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
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple, Type
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer,AutoProcessor,MambaModel,BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, AutoModel
from functools import partial
from utils_downstream.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc

torch.manual_seed(2024)
torch.cuda.empty_cache()

def eval_psnr(loader, model,vlm_model,processor,mamba_model,tokenizer,eval_type=None,device=None):
    model.eval()
    
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    

    mae,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()


    #for batch in loader:
    for step, (image, gt2D,img_1024_ori) in enumerate(loader):
       
        image, gt2D = image.to(device), gt2D.to(device)
        img_1024_ori = img_1024_ori.to(device)
        with torch.no_grad():
        
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs,output_hidden_states=True)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)
            
            ### Extract text information
            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
               mamba_outputs = mamba_model(**mamba_inputs)
               vision_outputs = vlm_model.vision_model(**vlm_inputs)
               image_features = vision_outputs.last_hidden_state[:,1:,:]
            
            text_features = mamba_outputs.last_hidden_state
            
            pred = torch.sigmoid(model(image,text_features,image_features))
            
            res = pred.squeeze().squeeze().cpu().numpy()
            gt = gt2D.squeeze().squeeze().cpu().numpy()
            
            mae.update(res, gt)
            sm.update(res,gt)
            #fm.update(res, gt)
            em.update(res,gt)
            wfm.update(res,gt)
            m_dice.update(res,gt)
            m_iou.update(res,gt)
            ber.update(res,gt)
        
        if pbar is not None:
            pbar.update(1)

    MAE = mae.show()
    #maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    ber = ber.show()
            

    if pbar is not None:
        pbar.close()

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


def _build_transforms():
    img_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    return img_transform, mask_transform


def _load_sample(img_path, gt_path, img_transform, mask_transform):
    img_pil = Image.open(img_path).convert('RGB')
    gt = Image.open(gt_path).convert('L')
    img_1024 = img_transform(img_pil)
    gt = mask_transform(gt)
    img_1024_ori_resized = img_pil.resize((1024, 1024), Image.BILINEAR)
    return (
        torch.tensor(img_1024).float(),
        torch.tensor(gt).long(),
        np.array(img_1024_ori_resized)
    )


class NpyDataset(Dataset):
    """
    Video dataset with nested subfolders.
    frames_root/<video>/<frame>.jpg  (all frames, e.g. 00000.jpg, 00001.jpg, ...)
    masks_root/<video>/<mask>.png   (1 every N frames, e.g. 00000.png, 00005.png, ...)
    Only trains on frames that have a corresponding mask (matched by stem).
    """
    def __init__(self, frames_root, masks_root, bbox_shift=20):
        self.bbox_shift = bbox_shift
        self.img_path_files = []
        self.gt_path_files  = []

        for video in sorted(os.listdir(masks_root)):
            mask_dir = join(masks_root, video)
            frame_dir = join(frames_root, video)
            if not os.path.isdir(mask_dir) or not os.path.isdir(frame_dir):
                continue
            # Build a stem -> mask path index
            mask_index = {
                os.path.splitext(f)[0]: join(mask_dir, f)
                for f in os.listdir(mask_dir) if f.endswith('.png')
            }
            for f in os.listdir(frame_dir):
                if not f.endswith('.jpg'):
                    continue
                stem = os.path.splitext(f)[0]
                if stem in mask_index:
                    self.img_path_files.append(join(frame_dir, f))
                    self.gt_path_files.append(mask_index[stem])

        self.img_path_files = sorted(self.img_path_files)
        self.gt_path_files  = sorted(self.gt_path_files)
        print(f"NpyDataset: {len(self.img_path_files)} frame-mask pairs found")
        self.img_transform, self.mask_transform = _build_transforms()

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):
        return _load_sample(
            self.img_path_files[index],
            self.gt_path_files[index],
            self.img_transform,
            self.mask_transform,
        )


class COD10KDataset(Dataset):
    """
    COD10K-v3 dataset.
    split: 'Train' or 'Test'
    Images:    <root>/<split>/Image/*.jpg
    GT Object: <root>/<split>/GT_Object/*.png
    """
    def __init__(self, root, split='Train'):
        img_dir = join(root, split, 'Image')
        gt_dir  = join(root, split, 'GT_Object')
        stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir) if f.endswith('.jpg')
        ])
        self.img_files = [join(img_dir, s + '.jpg') for s in stems]
        self.gt_files  = [join(gt_dir,  s + '.png') for s in stems]
        assert len(self.img_files) > 0, f"No images found in {img_dir}"
        print(f"COD10K {split}: {len(self.img_files)} samples")
        self.img_transform, self.mask_transform = _build_transforms()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        return _load_sample(
            self.img_files[index],
            self.gt_files[index],
            self.img_transform,
            self.mask_transform,
        )


class CAMODataset(Dataset):
    """
    CAMO dataset.
    split: 'Train' or 'Test'
    Images: <root>/Images/<split>/*.jpg
    GTs:    <root>/GT/<stem>.png  (single GT folder shared across splits)
    """
    def __init__(self, root, split='Train'):
        img_dir = join(root, 'Images', split)
        gt_dir  = join(root, 'GT')
        stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir) if f.endswith('.jpg')
        ])
        self.img_files = [join(img_dir, s + '.jpg') for s in stems]
        self.gt_files  = [join(gt_dir,  s + '.png') for s in stems]
        assert len(self.img_files) > 0, f"No images found in {img_dir}"
        print(f"CAMO {split}: {len(self.img_files)} samples")
        self.img_transform, self.mask_transform = _build_transforms()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        return _load_sample(
            self.img_files[index],
            self.gt_files[index],
            self.img_transform,
            self.mask_transform,
        )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/TrainDataset",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("--frames_path", type=str,
                    default="/Experiments/marcol01/frames_train",
                    help="Root with per-video subfolders of .jpg frames")
parser.add_argument("--masks_path", type=str,
                    default="/Experiments/marcol01/masks_train",
                    help="Root with per-video subfolders of .png masks")
parser.add_argument("--val_split", type=float, default=0.1,
                    help="Fraction of training data to use as validation (default: 0.1)")
# Dataset selection
parser.add_argument("--use_cod10k", action="store_true", default=False,
                    help="Include COD10K-v3 training set")
parser.add_argument("--use_camo", action="store_true", default=False,
                    help="Include CAMO training set")
parser.add_argument("--cod10k_path", type=str,
                    default="/Experiments/marcol01/COD10K-v3",
                    help="Root directory of the COD10K-v3 dataset")
parser.add_argument("--camo_path", type=str,
                    default="/Experiments/marcol01/CAMO",
                    help="Root directory of the CAMO dataset")
parser.add_argument("-task_name", type=str, default="....")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_h_4b8939.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default=".\\work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0002, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name)
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
        self.pseudo_mask_embed = nn.Sequential(
                nn.Conv2d(256, 256,3,1,1),
                nn.GELU())
       

    def forward(self, image,text_embeddings,image_features):
        blip_img_adap = image_features.reshape(1,-1,64,64)
        image_embedding = self.image_encoder(image, blip_img_adap)  # (B, 256, 64, 64)
      
        mamba_text = text_embeddings.reshape(1,-1,256)
        blip_img = image_features.reshape(1,-1,256)
        sparse_embeddings = torch.cat((mamba_text,blip_img),dim=1)

   
        
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
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    vlsam_model = VLSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
    ).to(device)

    for name, param in  vlsam_model.image_encoder.named_parameters():
        #print(name)
        if "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    vlsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in vlsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in vlsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(vlsam_model.image_encoder.parameters()) + list(
        vlsam_model.mask_decoder.parameters()
    )

    encoder_params = list(vlsam_model.image_encoder.parameters())
    decoder_params = list(vlsam_model.mask_decoder.parameters())
    embed_params = list(vlsam_model.pseudo_mask_embed.parameters())
    # Create parameter groups
    
    param_groups = [
    {'params': encoder_params, 'lr': args.lr * 0.1},
    {'params': decoder_params, 'lr': args.lr},
    {'params': embed_params, 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    
    lr_scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1.0e-6)
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    best_accuracy =0
    # Build training dataset(s)
    train_datasets = []
    if args.use_cod10k:
        train_datasets.append(COD10KDataset(args.cod10k_path, split='Train'))
    if args.use_camo:
        train_datasets.append(CAMODataset(args.camo_path, split='Train'))
    if not train_datasets:
        train_datasets.append(NpyDataset(args.frames_path, args.masks_path))

    from torch.utils.data import ConcatDataset, random_split
    full_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(2024)
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(test_dataset)} samples")

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            vlsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        
    #dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    #dino_backbone   = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").to(device)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    mamba_model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf").to(device)
    

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for step, (image, gt2D,img_1024_ori) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
                    
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)
            
            # Dino feature extraction
            #dino_inputs = dino_processor(img_1024_ori, return_tensors="pt").to(device)
            #with torch.no_grad():
            #    dino_outputs = dino_backbone(**dino_inputs)
            #    dino_features = dino_outputs.last_hidden_state[:,1:,:]  # (B, 64*64, 1024)

            ### Get sentence about the input image
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs,output_hidden_states=True)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)
            
            ### Extract text information
            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
               mamba_outputs = mamba_model(**mamba_inputs)
               
               vision_outputs = vlm_model.vision_model(**vlm_inputs)
               image_features = vision_outputs.last_hidden_state[:,1:,:]
            
            text_features = mamba_outputs.last_hidden_state

            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = vlsam_model(image, text_features, image_features)
                    loss = seg_loss(medsam_pred, gt2D.float()) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = vlsam_model(image,text_features,image_features)
                #medsam_pred = vlsam_model(image, text_features, dino_features)
                loss = seg_loss(medsam_pred, gt2D.float()) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
       
            epoch_loss += loss.item()
            iter_num += 1
        
        lr_scheduler.step()  # step once per epoch, not per batch
        
        
        result1, result2, result3, result4 = eval_psnr(test_dataloader, vlsam_model,vlm_model,processor,mamba_model,tokenizer,
                eval_type='cod',device=device)
        print({'Sm': result1})
        print({'Em': result2})
        print({'wFm': result3})
        print({'Mae': result4})

        epoch_loss /= step
        epoch_accuracy = (result1+result2+result3)/3
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": vlsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "vlsam_model_latest.pth"))
        ## save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            checkpoint = {
                "model": vlsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "vlsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
