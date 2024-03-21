# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen and Søren Hauberg, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from torch.distributions.kl import kl_divergence as KL
import torch.utils.data
from curve import compute_geodesic_dm
from tqdm import tqdm
import os

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.ContinuousBernoulli(logits=logits), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader)*epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = noise(x.to(device))
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1


def proximity(curve_points, latent):
    """
    Compute the average distance between points on a curve and a collection
    of latent variables.

    Parameters:
    curve_points: [torch.tensor]
        M points along a curve in latent space. tensor shape: M x latent_dim
    latent: [torch.tensor]
        N points in latent space (latent means). tensor shape: N x latent_dim

    The function returns a scalar.
    """
    pd = torch.cdist(curve_points, latent)  # M x N
    pd_min, _ = torch.min(pd, dim=0)
    pd_min_max = pd_min.max()
    return pd_min_max

class EnsembleVAE:
    def __init__(self, ensemble_path, device='cuda'):
        models = []
        for file in os.listdir(ensemble_path):
            path = os.path.join(ensemble_path, file)
            decoder = BernoulliDecoder(new_decoder())
            single_model = VAE(prior, decoder, encoder).to(device)
            single_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
            single_model.eval()
            models.append(single_model)
        
        self.models = models
    
    def encoder(self, x):
        return [model.encoder(x) for model in self.models]
    def decoder(self, z):
        return [model.decoder(z) for model in self.models]

    def exact_encoder(self, x):
        return sum([model.encoder(x).mean for model in self.models])/len(self.models)
    
    def decoder_entropy(self, z):
        return sum([model.decoder(z).entropy().mean().item() for model in self.models])/len(self.models)   

    def decoder_curve_energy(self, curve_points, N=10):
        """
        Use monte carlo approximation for expected value of KL of the ensemble
        
        """
        total_energy = 0
        for i in range(N):
            l, k = torch.randint(high=len(self.models), size=(2,))
            f_l = self.models[l].decoder
            f_k = self.models[k].decoder
            kl = KL(f_l(curve_points[1:]), f_k(curve_points[:-1]))
            total_energy += kl.sum()
        return total_energy # could divide by number of models here

if __name__ == "__main__":
    from torchvision import datasets, transforms
    import numpy as np
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'plot','train_ensemble'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--n_ensemble', type=int, default=1, help='Number of models in the ensemble')
    parser.add_argument('--plot', type=str, default='plot.png', help='file to save latent plot in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)
    
    num_train_data = 2048
    num_test_data = 16  # we keep this number low to only compute a few geodesics
    num_classes = 3
    train_tensors = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Define VAE model
    encoder = GaussianEncoder(encoder_net)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        
        os.makedirs(args.model)
        for i in range(args.n_ensembles):
            model = VAE(prior, BernoulliDecoder(new_decoder()), encoder).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train(model, optimizer, mnist_train_loader, args.epochs, args.device)

            # Save model
            model_path = os.path.join(args.model, f'{i}.pt')
            torch.save(model.state_dict(), model_path)

    elif args.mode == 'plot':
        import matplotlib.pyplot as plt

        model = EnsembleVAE(args.model, device=args.device)
        ## Load trained model

        ## Encode test and train data
        latents, labels = [], []
        with torch.no_grad():
            for x, y in mnist_train_loader:
                z = encoder(x.to(device))
                latents.append(z.mean)
                labels.append(y)
            latents = torch.concatenate(latents, dim=0).cpu()
            labels = torch.concatenate(labels, dim=0)

        range_: tuple[int, int] = (-8, 8)
        num_points: int = 50
        x, y = np.linspace(*range_, 50), np.linspace(*range_, 50)
        X, Y = np.meshgrid(x, y)

            ## Heatmap of entropy of the decoder
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                z = torch.tensor([X[i, j], Y[i, j]]).float().cuda()
                Z[i, j] = model.decoder_entropy(z)

        plt.pcolormesh(Z)
        plt.colorbar()

        ## Plot training data
        for k in range(num_classes):
            idx = labels == k
            plt.scatter(latents[idx, 0], latents[idx, 1], s=2)

        # Plot random geodesics
        num_curves: int = 3
        curve_indices = torch.randint(num_train_data, (num_curves, 2))  # (num_curves) x 2
        for k in range(num_curves):
            i = curve_indices[k, 0]
            j = curve_indices[k, 1]
            z0 = latents[i]
            z1 = latents[j]
            
            # z0 = latents[latents.argmin(dim=0)]
            # z1 = latents[latents.argmax(dim=0)]
            
            # TODO: Compute, and plot geodesic between z0 and z1
            ts = compute_geodesic_dm(z0, z1,energy_function=model.decoder_curve_energy, N_pieces=20, steps=100, lr=3e-4)
            plt.plot(ts.detach().numpy()[:,0], ts.detach().numpy()[:,1], color='r')

        plt.savefig(args.plot)

