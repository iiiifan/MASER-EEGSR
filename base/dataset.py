from os import path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mne.filter import notch_filter, filter_data
import warnings

class RawDataset(Dataset):
    def __init__(
            self,
            data_file,
            epoch_length,
            nchannels,
            sample_rate=None,
            notch_freq=None,
            low_pass=None,
            high_pass=None,
    ):
        if not path.exists(data_file):
            raise FileNotFoundError(f"Data file was not found: {data_file}")

        if notch_freq is not None or low_pass is not None or high_pass is not None:
            assert sample_rate is not None, (
                "sample rate must be specified to run a"
                "notch, low pass or high pass filter"
            )

        self.sample_rate = sample_rate
        self.notch_freq = notch_freq
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.nchannels = nchannels
        self.epoch_length = epoch_length
        # memory map the raw data
        name, nsamp = path.basename(data_file).split("-")[1].split("_")
        assert (
                "nsamp" in name
        ), "The file name does not contain the number of samples in the expected position."
        self.data = np.memmap(
            data_file,
            mode="r",
            dtype=np.float32,
            shape=(int(nsamp), epoch_length, nchannels),
        )
        self.indices = [ i for i in range(0, int(nsamp))]


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # read the current sample
        x = np.array(self.data[self.indices[idx]]).astype(float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            # notch filter
            if self.notch_freq is not None:
                x = notch_filter(
                    x.T,
                    self.sample_rate,
                    np.arange(self.notch_freq, self.sample_rate // 2, self.notch_freq),
                    verbose="warning",
                ).T

            # band pass filter
            if self.low_pass is not None or self.high_pass is not None:
                x = filter_data(
                    x.T,
                    self.sample_rate,
                    self.high_pass,
                    self.low_pass,
                    verbose="warning",
                ).T

        return (
            torch.from_numpy(x.astype(np.float32)),
        )


