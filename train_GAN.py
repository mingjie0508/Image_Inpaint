import os
from sklearn.model_selection import train_test_split
from InpaintDataset import InpaintDataset
from GAN import UNet, Discriminator
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss as mae
from config import *


def train_gan(models, dataloader, optimizers, epochs, checkpoint_paths, alpha=1.0):
    """
    Training loop for UNet-GAN. Performs training and saves final model.

    :param models: Tuple of Pytorch models, generator and discriminator.
    :param dataloader: Training dataloader that loads images and masks.
    :param optimizers: Tuple of Pytorch optimizers, for generator and discriminator.
    :param epochs: Number of epochs.
    :param checkpoint_paths: Tuple of output checkpoint paths.
    :param alpha: Float, hyperparameter that balances reconstruction loss with
        adversarial loss.
    :return: Tuple of list of training losses: reconstruction losses, 
        discriminator losses, and generator losses.
    """
    # unpack input
    generator, discriminator = models
    generator_optimizer, discriminator_optimizer = optimizers
    checkpoint_path_g, checkpoint_path_d = checkpoint_paths
    # losses for each epoch
    generator_losses, discriminator_losses, recon_losses = [], [], []
    for epoch in range(epochs):
        print('-'*10)
        print(f'Epoch {epoch + 1}/{epochs}:')
        generator.train()
        discriminator.train()
        generator_loss, discriminator_loss, recon_loss = 0.0, 0.0, 0.0
        for x, y in dataloader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            masked_x = x * y
            n = x.shape[0]
            ### Discriminator
            discriminator_optimizer.zero_grad()
            # pass real data through discriminator
            real_label = torch.full((n,), discriminator.REAL, dtype=torch.float, device=DEVICE)
            output = discriminator(x).view(-1)
            loss_d_real = discriminator.get_loss(output, real_label)
            loss_d_real.backward()
            # generate fake data
            pred_x = generator(masked_x)
            fake_x = masked_x + (1-y) * pred_x
            # pass fake data through discriminator
            fake_label = torch.full((n,), discriminator.FAKE, dtype=torch.float, device=DEVICE)
            output = discriminator(fake_x.detach()).view(-1)
            loss_d_fake = discriminator.get_loss(output, fake_label)
            loss_d_fake.backward()
            # compute error for discriminator
            loss_d = loss_d_real + loss_d_fake
            discriminator_loss += loss_d.item()
            # update discriminator
            discriminator_optimizer.step()
            ### Generator
            generator_optimizer.zero_grad()
            # pass fake data through generator and discriminator
            output = discriminator(fake_x).view(-1)
            loss_g_fake = discriminator.get_loss(output, real_label)
            loss_enc = alpha * generator.get_loss(pred_x, x, y)
            loss_g = loss_g_fake + loss_enc
            loss_g.backward()
            # compute error for generator
            generator_loss += loss_g_fake.item()
            recon_loss += loss_enc.item()
            # update generator
            generator_optimizer.step()

        epoch_recon_loss = recon_loss / len(dataloader)
        epoch_generator_loss = generator_loss / len(dataloader)
        epoch_discriminator_loss = discriminator_loss / len(dataloader) / 2
        recon_losses.append(epoch_recon_loss)
        generator_losses.append(epoch_generator_loss)
        discriminator_losses.append(epoch_discriminator_loss)
        print('Train recon loss:', round(epoch_recon_loss, 4))
        print('Train generator loss:', round(epoch_generator_loss, 4))
        print('Train discriminator loss:', round(epoch_discriminator_loss, 4))
        
    # save final models
    torch.save(
        {'epoch': epoch,
         'model_state_dict': generator.state_dict(),
         'optimizer_state_dict': generator_optimizer.state_dict(),
         'loss': epoch_generator_loss},
        checkpoint_path_g
    )
    torch.save(
        {'epoch': epoch,
         'model_state_dict': discriminator.state_dict(),
         'optimizer_state_dict': discriminator_optimizer.state_dict(),
         'loss': epoch_discriminator_loss},
        checkpoint_path_d
    )
    return generator_losses, discriminator_losses, recon_losses


def test_gan(models, dataloader, alpha=1.0):
    """
    Test loop for UNet-GAN. Prints test loss and evaluation metrics.
    Returns predicted images.

    :param models: Tuple of Pytorch models, generator and discriminator.
    :param dataloader: Test dataloader that loads images and masks.
    :param alpha: Float, hyperparameter that balances reconstruction loss with
        adversarial loss.
    :return: List of Tensors, predicted images.
    """
    # unpack input
    generator, discriminator = models
    generator.eval()
    discriminator.eval()

    running_generator_loss = 0.0
    running_discriminator_loss = 0.0
    running_recon_loss = 0.0
    running_mae = 0.0
    pred_list = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            masked_x = x * y
            n = x.shape[0]
            ### Discriminator
            # pass real data through discriminator
            real_label = torch.full((n,), discriminator.REAL, dtype=torch.float, device=DEVICE)
            output = discriminator(x).view(-1)
            loss_d_real = discriminator.get_loss(output, real_label)
            # generate fake data
            pred_x = generator(masked_x)
            fake_x = masked_x + (1-y) * pred_x
            # pass fake data through discriminator
            fake_label = torch.full((n,), discriminator.FAKE, dtype=torch.float, device=DEVICE)
            output = discriminator(fake_x.detach()).view(-1)
            loss_d_fake = discriminator.get_loss(output, fake_label)
            # compute error for discriminator
            loss_d = loss_d_real + loss_d_fake
            running_discriminator_loss += loss_d.item()
            ### Generator
            # pass fake data through generator and discriminator
            output = discriminator(fake_x).view(-1)
            loss_g_fake = discriminator.get_loss(output, real_label)
            loss_enc = alpha * generator.get_loss(pred_x, x, y)
            # compute error for generator
            running_generator_loss += loss_g_fake.item()
            running_recon_loss += loss_enc.item()

            running_mae += mae(fake_x, x).item()
            pred_list.append(pred_x.cpu())

    pred_list = torch.cat(pred_list)
    recon_loss = running_recon_loss / len(dataloader)
    generator_loss = running_generator_loss / len(dataloader)
    discriminator_loss = running_discriminator_loss / len(dataloader) / 2
    mae_score = running_mae / len(dataloader)
    print('Test recon loss:', round(recon_loss, 4))
    print('Test generator loss:', round(generator_loss, 4))
    print('Test discriminator loss:', round(discriminator_loss, 4))
    print('Test MAE:', round(mae_score, 4))
    return pred_list


if __name__ == "__main__":
    # get the list of all image paths
    paths = [os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH)]
    # split paths into training and testing
    paths_train, paths_test = train_test_split(
        paths, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    # configure datasets
    train_dataset = InpaintDataset(paths_train, IMG_TRANSFORM, MASK_TRANSFORM)
    test_dataset = InpaintDataset(paths_test, IMG_TRANSFORM, MASK_TRANSFORM)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # alpha hyperparameter in loss
    # we use alpha = 1.2 and alpha = 1.0
    ALPHA = 1.0

    # configure generator
    generator = UNet(in_channels=3, out_channels=3)
    generator = generator.to(DEVICE)
    discriminator = Discriminator(in_channels=3)
    discriminator = discriminator.to(DEVICE)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=1e-4, weight_decay=0.0
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, weight_decay=0.0
    )

    # configure discriminator
    generator_path = f"./models/generator_alpha{ALPHA:.1f}.pth"
    discriminator_path = "./models/discriminator_alpha{ALPHA:.1f}.pth"
    checkpoint = torch.load(generator_path)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    checkpoint = torch.load(discriminator_path)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # train Unet-GAN
    epochs = 9  # can modify
    gan_generator_loss, gan_discriminator_loss, gan_recon_loss = train_gan(
        (generator, discriminator), 
        train_dataloader, 
        (generator_optimizer, discriminator_optimizer),
        epochs=epochs,
        checkpoint_paths=(generator_path, discriminator_path),
        alpha=ALPHA
    )

    # test UNet-GAN
    print(f"Testing UNet-GAN, alpha = {ALPHA:.1f}")
    gan_pred = test_gan((generator, discriminator), test_dataloader, alpha=ALPHA)
