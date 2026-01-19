import os, torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import PathPlanningDataset
from generator import ViTGenerator
from discriminator import ViTDiscriminator
from criterion import GANLoss, WeightedL1Loss, get_weight_map

# -------Hyperparameters-------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
BETA1, BETA2 = 0.5, 0.999
EPOCHS = 100
LAMBDA_L1 = 100
SAVE_DIR = "checkpoints"
DATA_DIR = "datasets"

def train():
    print(f"Using device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = PathPlanningDataset(root_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    netG = ViTGenerator().to(DEVICE)
    netD = ViTDiscriminator().to(DEVICE)
    
    criterion_GAN = GANLoss().to(DEVICE)
    criterion_L1 = WeightedL1Loss().to(DEVICE)

    optimizer_G = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    for epoch in range(1, EPOCHS + 1):
        netG.train()
        netD.train()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}")