import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_Encoder(nn.Module):


    def __init__(self, input_dim, latent_dim, hidden_dim=32):
        super(VAE_Encoder, self).__init__()

        self.tanh = nn.Tanh()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:

                    if m == self.FC_mean:
                        nn.init.normal_(m.bias.data, mean=0.0, std=0.1)

                    elif m == self.FC_var:
                        nn.init.normal_(m.bias.data, mean=-2.0, std=0.1)
                    else:
                        m.bias.data.zero_()

    def forward(self, x):
        x_hidden = self.tanh(self.hidden(x))
        mean = self.FC_mean(x_hidden)
        log_var = self.FC_var(x_hidden)
        return mean, log_var


class VAE_Decoder(nn.Module):


    def __init__(self, latent_dim, output_dim, hidden_dim=32):
        super(VAE_Decoder, self).__init__()

        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.FC = nn.Linear(hidden_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_hidden = self.tanh(self.hidden(x))
        x = self.FC(x_hidden)
        return x


class VAE_Network(nn.Module):


    def __init__(self, input_dim, latent_dim, output_dim, beta=1.0, log_alpha=100.0,
                 hidden_dim=32):


        super(VAE_Network, self).__init__()

        self.beta = beta
        self.log_alpha = log_alpha

        self.Encoder = VAE_Encoder(input_dim, latent_dim, hidden_dim)
        self.Decoder = VAE_Decoder(latent_dim, output_dim, hidden_dim)

    def reparameterization(self, mean, log_var):


        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        z = mean + std * epsilon
        return z

    def forward(self, x):


        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite cross-sectional returns")


        x_transformed = x


        mean, log_var = self.Encoder(x_transformed)

        log_var = torch.clamp(log_var, -10.0, 10.0)

        z = self.reparameterization(mean, log_var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var

    def get_factor_returns(self, x, use_mean: bool = False):


        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite cross-sectional returns")


        x_transformed = x


        mean, log_var = self.Encoder(x_transformed)

        log_var = torch.clamp(log_var, -10.0, 10.0)

        if use_mean:

            F_t = mean
        else:

            F_t = self.reparameterization(mean, log_var)
        return F_t

    def loss_function(self, x, x_hat, mean, log_var):


        if not torch.isfinite(x).all() or not torch.isfinite(x_hat).all():

            raise FloatingPointError("Nonfinite VAE inputs or reconstruction")


        x_transformed = x


        log_var_clamped = torch.clamp(log_var, -10, 10)


        reproduction_loss = torch.nn.functional.mse_loss(x_hat, x_transformed, reduction='mean')


        KLD = -0.5 * torch.mean(1 + log_var_clamped - mean.pow(2) - log_var_clamped.exp())


        KLD = torch.clamp(KLD, min=0.0)

        total_loss = reproduction_loss + self.beta * KLD


        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise FloatingPointError("Nonfinite VAE objective")

        return total_loss
