import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datasets
import losses
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :]


class Block(nn.Module):
    def __init__(self, in_chan, out_chan, policy_in, policy_out, downsample=True, delineate_dims=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_chan, in_chan, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(0.2)
        )
        self.policy = nn.Sequential(
            nn.Linear(policy_in, policy_in),
            nn.LeakyReLU(0.2),
            nn.Linear(policy_in, policy_in),
            nn.LeakyReLU(0.2)
        )

        self.delineate = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(delineate_dims),
            nn.LeakyReLU(0.2)
        )

        self.mash = nn.Sequential(
            nn.Linear(policy_in + delineate_dims, (policy_in + policy_out + delineate_dims) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((policy_in + policy_out + delineate_dims) // 3, policy_out),
            nn.LeakyReLU(0.2)
        )

        self.trans = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, (3, 3), padding=1, stride=2),
            nn.LeakyReLU(0.2)
        ) if downsample else nn.Sequential(
            nn.Conv2d(in_chan, out_chan, (3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, policy, conv):
        policy = self.policy(policy) + policy
        conv = self.conv(conv) + conv

        return self.mash(torch.cat([self.delineate(conv), policy], dim=1)), self.trans(conv)


class ImageConditionedLinearUNet(nn.Module):
    def __init__(self, image_size, out_size, max_times=200, positional_features=18,
                 conv_features=(8, 16, 32, 64, 128, 256),
                 lin_features=(32, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.LazyConv2d(conv_features[0], (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(conv_features[0])
        )

        self.forward_blocks = nn.ModuleList([
            Block(icf, ocf, ipf, opf, downsample=True) for icf, ocf, ipf, opf in zip(conv_features[:-1],
                                                                                     conv_features[1:],
                                                                                     lin_features[:-1],
                                                                                     lin_features[1:])
        ])

        self.down_blocks = nn.ModuleList([
            Block(icf, ocf, ipf, opf, downsample=False) for icf, ocf, ipf, opf in reversed(list(zip(
                conv_features[1:],
                conv_features[:-1],
                lin_features[1:],
                lin_features[:-1]
            )))
        ])

        self.embeddings = PositionalEncoding(max_times, positional_features)
        self.fc0 = nn.Sequential(
            nn.Linear(lin_features[0], (lin_features[0] + out_size) // 2),
            nn.LeakyReLU(0.2),
            nn.Linear((lin_features[0] + out_size) // 2, out_size),
            #nn.Sigmoid()
        )

        self.out_size = out_size

    def forward(self, policy, image, time, verbose=0):
        time_embeddings = self.embeddings(time)
        policy = torch.cat([policy, time_embeddings], dim=1)
        image = self.conv1(image)
        hist = {}

        for i, block in enumerate(self.forward_blocks):
            if verbose > 0:
                print(i, policy.shape, image.shape)

            hist[i] = policy, image
            policy, image = block(policy, image)

        hist[len(self.forward_blocks)] = policy, image

        if verbose > 0:
            print(len(self.forward_blocks), policy.shape, image.shape)
            print("back")

        for i, block in enumerate(self.down_blocks):
            if verbose > 0:
                print(i, policy.shape, image.shape)

            res_pol, res_image = hist[len(self.down_blocks) - i]
            policy, image = block(policy + res_pol, image + res_image)

        return self.fc0(policy)


class Hyperparameters(object):
    def __init__(self, times, beta_start=0.0001, beta_end=0.2):
        self.betas = torch.linspace(beta_start, beta_end, times)
        self.alphas = 1 - self.betas

        self.beta_cumprods = torch.cumprod(self.betas, dim=0)
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0)

        self.T = times


class StableDiffusionLearner(nn.Module):
    def __init__(self, in_dimensions, latent_dimensions, autoencoder_in_latent_controller,
                 diffusion_model, hyperparameters):
        super().__init__()

        self.in_channels = in_dimensions[0]
        self.img_size = in_dimensions[1:]

        self.latent_dimensions = latent_dimensions

        self.autoencoder = autoencoder_in_latent_controller
        self.diffusion_model = diffusion_model

        self.perceptual_loss = losses.PerceptualLoss()
        self.hyperparameters = hyperparameters
        self.mse = nn.MSELoss()

    def get_losses(self, images, gt_classes, t):
        noise = torch.randn_like(gt_classes.type(torch.float32))
        gt_classes_T = gt_classes * self.hyperparameters.alpha_cumprods[t].unsqueeze(1) + \
                       noise * self.hyperparameters.beta_cumprods[t].unsqueeze(1)

        pred_noise = self.diffusion_model(gt_classes_T, images, t)
        loss = self.mse(pred_noise, noise)

        return loss

    @torch.no_grad()
    def generate(self, images):
        policies = torch.ones(images.shape[0], self.diffusion_model.out_size) / 2
        for t in range(self.hyperparameters.T):
            pred_noise = self.diffusion_model(policies, images, torch.zeros(images.shape[0])+t)
            policies = policies - pred_noise

        return policies

    def train_autoencoder(self, dataset, iterations):
        for i in range(iterations):
            for images in dataset:
                self.autoencoder.train(images)


def test_dmfusion():
    learner = StableDiffusionLearner((3, 256, 256), (3, 256, 256), None,
                                     ImageConditionedLinearUNet((3, 256, 256), 14, positional_features=18,
                                                                ), Hyperparameters(200))
    x = torch.randn(12, 3, 256, 256)
    policy = torch.randn(12, 14)
    gt_classes = torch.randint(0, 1, (12, 14), dtype=torch.float32)

    time = torch.randint(0, 200, (12,))
    learner.diffusion_model(policy, x, time, verbose=1)

    optimizer = optim.Adam(learner.diffusion_model.parameters(), lr=1e-4)
    for _ in range(1):
        optimizer.zero_grad()
        time = torch.randint(0, 200, (12,))

        loss = learner.get_losses(x, gt_classes, time)
        print(loss.item())
        loss.backward()

        optimizer.step()


if __name__ == "__main__":
    test_dmfusion()

