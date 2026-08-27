"""PyTorch datasets for online training and fixed validation generation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .simulation import TrainingSample, generate_training_sample


TensorSample = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def sample_to_tensors(sample: TrainingSample) -> TensorSample:
    """Convert the generator's arrays to the float32 training contract."""

    return (
        torch.tensor(sample.input_spectra, dtype=torch.float32),
        torch.tensor(sample.isotope_distribution, dtype=torch.float32),
        torch.tensor(sample.lithium6_percentage, dtype=torch.float32),
        torch.tensor(sample.targets, dtype=torch.float32),
    )


class OnlineSyntheticDataset(Dataset[TensorSample]):
    """Generate a fresh synthetic Li--Ne spectrum for every item request."""

    def __init__(
        self,
        number_of_samples: int,
        *,
        output_length: int = 512,
        number_of_isotope_bins: int = 300,
        normalization_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.number_of_samples = int(number_of_samples)
        self.output_length = int(output_length)
        self.number_of_isotope_bins = int(number_of_isotope_bins)
        self.normalization_range = tuple(map(float, normalization_range))

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, index: int) -> TensorSample:
        del index
        return sample_to_tensors(
            generate_training_sample(
                output_length=self.output_length,
                number_of_isotope_bins=self.number_of_isotope_bins,
                normalization_range=self.normalization_range,
                random_parameters=True,
            )
        )


class PregeneratedValidationDataset(Dataset[TensorSample]):
    """Generate validation spectra once and retain them in memory."""

    def __init__(
        self,
        number_of_samples: int,
        *,
        output_length: int = 512,
        number_of_isotope_bins: int = 300,
        normalization_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.samples = [
            sample_to_tensors(
                generate_training_sample(
                    output_length=output_length,
                    number_of_isotope_bins=number_of_isotope_bins,
                    normalization_range=normalization_range,
                    random_parameters=True,
                )
            )
            for _ in range(int(number_of_samples))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TensorSample:
        return self.samples[index]


def collate_training_samples(batch: Sequence[TensorSample]) -> TensorSample:
    """Stack the four tensors used by the final training objective."""

    fields = zip(*batch, strict=True)
    return tuple(torch.stack(list(field), dim=0) for field in fields)  # type: ignore[return-value]


def make_synthetic_dataloader(
    *,
    number_of_samples: int,
    batch_size: int,
    output_length: int = 512,
    number_of_isotope_bins: int = 300,
    normalization_range: tuple[float, float] = (0.0, 1.0),
    shuffle: bool = True,
    number_of_workers: int = 8,
    prefetch_factor: int = 4,
    pregenerate: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader[TensorSample]:
    """Build the online-training or fixed-validation loader."""

    dataset_type = PregeneratedValidationDataset if pregenerate else OnlineSyntheticDataset
    dataset = dataset_type(
        number_of_samples,
        output_length=output_length,
        number_of_isotope_bins=number_of_isotope_bins,
        normalization_range=normalization_range,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_of_workers,
        collate_fn=collate_training_samples,
        pin_memory=True,
        prefetch_factor=prefetch_factor if number_of_workers > 0 else None,
        persistent_workers=number_of_workers > 0,
        generator=generator,
    )


__all__ = [
    "OnlineSyntheticDataset",
    "PregeneratedValidationDataset",
    "TensorSample",
    "collate_training_samples",
    "make_synthetic_dataloader",
    "sample_to_tensors",
]
