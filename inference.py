import torch
import argparse
import yaml
import pickle
from torch.utils.data import DataLoader
from base.ssm import MambaBlock
from base.modelMM import Maser

def build_model(args):
    enc = MambaBlock(
        sample_times=args.sample_times,
        elec_channels=args.elec_channels,
        patch_size=(args.patch_size, 1),
        depth=args.mamba_depth,
        d_state=args.d_state
    )
    dec = MambaBlock(
        sample_times=args.sample_times,
        elec_channels=args.elec_channels,
        patch_size=(args.patch_size, 1),
        depth=args.mamba_depth,
        d_state=args.d_state
    )
    maser = Maser(
        lr_extractor=enc,
        hr_predictor=dec,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        device=args.device
    ).to(args.device)
    return maser


def load_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)

    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    return model, args


def load_test_data(data_path, batch_size):
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return test_loader


def inference(model, data_loader, unmask_list, args):
    total_loss, total_nmse, total_ppc = 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for signals in data_loader:
            eegs = signals[0].unsqueeze(1).to(args.device)
            loss, nmse, ppc = model(eegs, unmask_list, test_flag=True)

            total_loss += loss.item()
            total_nmse += nmse.item()
            total_ppc += ppc.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_nmse = total_nmse / num_batches
    avg_ppc = total_ppc / num_batches

    return avg_loss, avg_nmse, avg_ppc


if __name__ == "__main__":
    config_path = 'config/case2-4x-MM-state8-Mdep2.yml'
    checkpoint_path = 'ckpt/last.ckpt'
    test_data_path = 'data/test_data.dat'

    model, args = load_model(config_path, checkpoint_path)
    test_loader = load_test_data(test_data_path, args.batch_size)

    unmask_list = args.unmasked_list

    loss, nmse, ppc = inference(model, test_loader, unmask_list, args)
    print(f"Inference results - Loss: {loss:.4f}, NMSE: {nmse:.4f}, PPC: {ppc:.4f}")