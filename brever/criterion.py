from itertools import permutations

import torch


eps = torch.finfo(torch.float32).eps


class SISNR:
    def __call__(self, data, target, lengths):
        """
        Calculate SI-SNR with PIT.

        Parameters
        ----------
        data: tensor
            Estimated sources. Shape `(batch_size, sources, length)`
        target: tensor
            True sources. Shape `(batch_size, sources, length)`

        Returns
        -------
        si_snr : float
            SI-SNR.
        """
        # (B, S, L) = (batch_size, sources, length)

        # apply mask a first time to get correct normalization statistics
        data, target = apply_mask(data, target, lengths)

        # normalize
        lengths = torch.as_tensor(lengths, device=data.device).view(-1, 1, 1)
        data = data - data.sum(dim=2, keepdim=True)/lengths
        target = target - target.sum(dim=2, keepdim=True)/lengths

        # apply mask a second time since trailing samples are now non-zero
        data, target = apply_mask(data, target, lengths)

        # calculate pair-wise snr
        s_hat = torch.unsqueeze(data, dim=1)  # (B, 1, S, L)
        s = torch.unsqueeze(target, dim=2)  # (B, S, 1, L)
        s_target = torch.sum(s_hat*s, dim=3, keepdim=True)*s \
            / torch.sum(s**2, dim=3, keepdim=True)  # (B, S, S, L)
        e_noise = s_hat - s_target  # (B, S, S, L)
        si_snr = torch.sum(s_target**2, dim=3) \
            / (torch.sum(e_noise ** 2, dim=3) + eps)  # (B, S, S, L)
        si_snr = 10*torch.log10(si_snr + eps)

        # permute
        S = data.shape[1]
        perms = data.new_tensor(list(permutations(range(S))), dtype=torch.long)
        index = torch.unsqueeze(perms, 2)
        one_hot = data.new_zeros((*perms.size(), S)).scatter_(2, index, 1)
        snr_set = torch.einsum('bij,pij->bp', [si_snr, one_hot])
        max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
        max_snr /= S
        loss = 0 - torch.mean(max_snr)
        return loss


class SNR:
    def __call__(self, data, target, lengths):
        """
        Calculate SNR without PIT.

        Parameters
        ----------
        data: tensor
            Estimated sources. Shape `(batch_size, sources, length)`.
        target: tensor
            True sources. Shape `(batch_size, sources, length)`

        Returns
        -------
        snr : float
            SNR.
        """
        # (B, S, L) = (batch_size, sources, length)
        data, target = apply_mask(data, target, lengths)
        snr = torch.sum(target**2, dim=-1) \
            / (torch.sum((target-data)**2, dim=-1) + eps)  # (B, S)
        snr = 10*torch.log10(snr + eps)  # (B, S)
        loss = -torch.mean(snr)
        return loss


class MSE:
    def __call__(self, data, target, lengths):
        data, target = apply_mask(data, target, lengths)
        lengths = torch.as_tensor(lengths, device=data.device).view(-1, 1)
        loss = (data-target).pow(2).sum(-1)/lengths
        return loss.mean()


def get_criterion(name):
    if name == 'SISNR':
        return SISNR()
    elif name == 'SNR':
        return SNR()
    elif name == 'MSE':
        return MSE()
    else:
        raise ValueError(f'Unrecognized criterion, got {name}')


def apply_mask(data, target, lengths):
    assert len(lengths) == data.size(0)
    mask = torch.zeros(*data.shape, device=data.device)
    for i, length in enumerate(lengths):
        mask[i, ..., :length:] = 1
    return data*mask, target*mask
