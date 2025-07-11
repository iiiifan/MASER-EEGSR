import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from scipy.stats import pearsonr


class Maser(nn.Module):
    def __init__(self, *, lr_extractor, hr_predictor, decoder_dim=320, decoder_depth=2,
                 decoder_heads=8, decoder_dim_head=64, device='cpu'):
        super().__init__()
        self.device = device
        self.lr_extractor = lr_extractor
        self.hr_predictor = hr_predictor
        num_patches, encoder_dim = lr_extractor.pos_embedding.shape[-2:]

        self.to_patch = lr_extractor.to_patch_embedding[:1]
        self.to_eeg = lr_extractor.reverse_img[:1]
        self.token_embedding = nn.Linear(160, 512)
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        # self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads,
        #                            dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)


    def __str__(self):
        return f"""MASER(
        (to_patch): {self.to_patch}
        (mask_token): {self.mask_token.shape}
        (lr_extractor): {self.lr_extractor}
        (enc_to_dec): {self.enc_to_dec}
        (decoder_pos_emb): {self.decoder_pos_emb}
        (hr_predictor): {self.hr_predictor}
        )
        """

    def forward(self, img, unmasked_list, test_flag=False):
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        unmasked_list_new = []
        for num in unmasked_list:
            unmasked_list_new.append(num * 2)
            unmasked_list_new.append(num * 2+1)
        masked_list = [x for x in range(0, 64 * 2) if x not in unmasked_list_new]
        num_masked = len(masked_list)

        masked_indices = torch.tensor([masked_list] * batch).to(self.device)
        unmasked_indices = torch.tensor([unmasked_list_new] * batch).to(self.device)

        batch_range = torch.arange(batch, device=self.device)[:, None]
        unmasked_patches = patches[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]

        encoded_tokens = self.lr_extractor(unmasked_patches, x_size=(unmasked_patches.shape[1], 1))
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # self.mask_token.data = unmasked_tokens.mean(dim=[0, 1])
        mean_token2 = torch.mean(torch.mean(unmasked_patches, dim=1), dim=0)
        mask_tokens2 = repeat(mean_token2, 'd -> b n d', b=batch, n=num_masked)
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        mean_token = torch.mean(torch.mean(decoder_tokens, dim=1), dim=0)
        mask_tokens = repeat(mean_token, 'd -> b n d', b=batch, n=num_masked)
        # Interpolate the original patches
        mask_tokens = torch.add(mask_tokens, mask_tokens2)
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)

        # decoder_tokens = torch.cat()
        decoded_tokens = self.hr_predictor(decoder_tokens, x_size=(decoder_tokens.shape[1], 1))
        mask_tokens = decoded_tokens[:, :num_masked]
        # decoded_tokens[:, unmasked_list_new] = patches[:, unmasked_list_new]
        pred_pixel_values = mask_tokens
        pre_patch = patches.detach().clone()
        pre_patch[:, masked_list] = decoded_tokens[:, :num_masked]

        pre_patch = self.to_eeg(pre_patch).squeeze(1).transpose(1, 2)
        patches = self.to_eeg(patches).squeeze(1).transpose(1, 2)
        mse_loss = F.mse_loss(pre_patch, patches)
        smoothness_loss = torch.mean(torch.abs(decoded_tokens[:, 1:] - decoded_tokens[:, :-1]))
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        recon_loss = mse_loss + 0.01 * smoothness_loss + 0.00001 * l2_reg

        if test_flag:
            pre_patch_np = pre_patch.cpu().numpy()
            patches_np = patches.cpu().numpy()
            ppc, _ = pearsonr(pre_patch_np.flatten(), patches_np.flatten())
            var = patches.var()
            nmse = mse_loss / var
            return recon_loss, nmse, ppc

        return recon_loss