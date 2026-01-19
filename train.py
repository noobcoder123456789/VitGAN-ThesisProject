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
LAMDBA_ADV = 1
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
        for i, (condition, target) in pbar:
            condition = condition.to(DEVICE)
            target = target.to(DEVICE)

            # -------Train Discriminator-------
            optimizer_D.zero_grad()

            pred_real = netD(condition, target)
            loss_D_real = criterion_GAN(pred_real, True)

            with torch.no_grad():
                fake_image = netG(condition)

            pred_fake = netD(condition, fake_image.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)

            loss_D = 0.5 * (loss_D_fake + loss_D_real)
            loss_D.backward()
            optimizer_D.step()

            # -------Train Generator-------
            optimizer_G.zero_grad()

            fake_image_new = netG(condition)
            pred_fake_new = netD(condition, fake_image_new)
            loss_G_Adv = criterion_GAN(pred_fake_new, True)

            weights = get_weight_map(condition, DEVICE, penalty_weight=50.0)
            loss_G_L1 = criterion_L1(fake_image_new, target, weights)

            loss_G = LAMBDA_L1 * loss_G_L1 + LAMDBA_ADV * loss_G_Adv
            loss_G.backward()
            optimizer_G.step()

            # -------LOGGING-------
            pbar.set_postfix({
                'Loss_D': f"{loss_D.item():.4f}",
                'Loss_G': f"{loss_G.item():.4f}",
                'L1': f"{loss_G_L1.item():.4f}",
            })

        if epoch % 5 == 0:
            torch.save(netG.state_dict(), f"{SAVE_DIR}/netG_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"{SAVE_DIR}/netD_epoch_{epoch}.pth")
            print(f"Saved model at epoch {epoch}!")

if __name__ == "__main__":
    train()