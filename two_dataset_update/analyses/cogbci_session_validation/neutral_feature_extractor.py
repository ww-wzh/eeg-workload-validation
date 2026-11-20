# Extracted unchanged from the first-dataset neutral feature extractor.
# Only definitions required by this external validation are included.
from __future__ import annotations
import math
import json
RUNNER_VERSION = "COGBCI-session-v1"
from dataclasses import dataclass
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    normalization: str
    line_attenuation_50hz: float
    frontal_multiplier: float
    theta_alpha_multiplier: float

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "variant_id": self.variant_id,
                "normalization": self.normalization,
                "line_attenuation_50hz": self.line_attenuation_50hz,
                "frontal_multiplier": self.frontal_multiplier,
                "theta_alpha_multiplier": self.theta_alpha_multiplier,
                "runner_version": RUNNER_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def extract_features(
    eeg: np.ndarray,
    *,
    sfreq: float,
    win_sec: float,
    stride_sec: float,
    bands: list[tuple[float, float]],
    config: VariantConfig,
    device: torch.device,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the frozen 272-D extractor with one explicit variant configuration."""
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[1] != 8:
        raise ValueError(f"Expected raw EEG shaped (samples, 8), got {eeg.shape}")

    nyquist = float(sfreq) / 2.0
    if not (0.5 > 0.0 and 55.0 < nyquist):
        raise ValueError(f"Sampling rate {sfreq} does not support the frozen 0.5-55 Hz bandpass")
    sos = butter(4, [0.5 / nyquist, 55.0 / nyquist], btype="band", output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=0).astype(np.float32)

    if config.normalization == "recording_zscore":
        mean = eeg.mean(axis=0, keepdims=True)
        std = np.clip(eeg.std(axis=0, keepdims=True), 1e-6, None)
        eeg = ((eeg - mean) / std).astype(np.float32)
    elif config.normalization not in {"window_zscore", "none"}:
        raise ValueError(f"Unsupported normalization mode: {config.normalization}")

    win_samples = int(round(float(win_sec) * float(sfreq)))
    stride_samples = int(round(float(stride_sec) * float(sfreq)))
    if win_samples != 2125 or stride_samples != 125:
        raise AssertionError(
            f"Frozen sampling geometry expected 2125/125 samples, got {win_samples}/{stride_samples}"
        )
    if len(eeg) < win_samples:
        raise ValueError("Recording is shorter than one frozen analysis window")
    n_windows = 1 + (len(eeg) - win_samples) // stride_samples
    n_fft = next_power_of_two(win_samples)
    if n_fft != 4096:
        raise AssertionError(f"2125-sample windows must use a 4096-point FFT, got {n_fft}")

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sfreq))
    band_masks = [(freqs >= low) & (freqs < high) for low, high in bands]
    hann = torch.from_numpy(np.hanning(win_samples).astype(np.float32)).to(device)
    eps = 1e-12
    feature_parts: list[np.ndarray] = []
    covariance_parts: list[np.ndarray] = []
    starts = [index * stride_samples for index in range(n_windows)]

    for chunk_start in range(0, n_windows, int(chunk_size)):
        chunk_starts = starts[chunk_start : chunk_start + int(chunk_size)]
        chunk = np.stack(
            [eeg[start : start + win_samples].T for start in chunk_starts],
            axis=0,
        ).astype(np.float32)
        x_raw = torch.from_numpy(chunk).to(device)
        if config.normalization == "window_zscore":
            channel_mean = x_raw.mean(dim=-1, keepdim=True)
            channel_std = x_raw.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
            x = (x_raw - channel_mean) / channel_std
        else:
            x = x_raw

        windowed = x * hann[None, None, :]
        spectrum = torch.fft.rfft(windowed, n=n_fft, dim=-1)
        power = (spectrum.real**2 + spectrum.imag**2) / float(n_fft)

        if config.line_attenuation_50hz < 1.0:
            notch_mask = (freqs >= 48.5) & (freqs <= 51.5)
            notch_tensor = torch.from_numpy(notch_mask.astype(np.bool_)).to(device)
            power[:, :, notch_tensor] *= float(config.line_attenuation_50hz)

        total_mask = (freqs >= 0.5) & (freqs < 55.0)
        total_tensor = torch.from_numpy(total_mask.astype(np.bool_)).to(device)
        total_power = power[:, :, total_tensor].mean(dim=-1, keepdim=True).clamp_min(eps)

        absolute_bands = []
        relative_bands = []
        for mask in band_masks:
            mask_tensor = torch.from_numpy(mask.astype(np.bool_)).to(device)
            band_power = power[:, :, mask_tensor].mean(dim=-1)
            absolute_bands.append(torch.log(band_power + eps))
            relative_bands.append(band_power / total_power.squeeze(-1))
        absolute_band_power = torch.stack(absolute_bands, dim=-1)
        relative_band_power = torch.stack(relative_bands, dim=-1)

        if abs(config.frontal_multiplier - 1.0) > 1e-6:
            frontal_indices = torch.arange(
                0,
                relative_band_power.shape[1] // 2,
                device=device,
                dtype=torch.long,
            )
            log_weight = float(np.log(config.frontal_multiplier))
            absolute_band_power = absolute_band_power.clone()
            relative_band_power = relative_band_power.clone()
            absolute_band_power[:, frontal_indices, :] += log_weight
            relative_band_power[:, frontal_indices, :] *= float(config.frontal_multiplier)

        if config.theta_alpha_multiplier > 1.0:
            relative_band_power = relative_band_power.clone()
            relative_band_power[:, :, 1] *= float(config.theta_alpha_multiplier)
            relative_band_power[:, :, 2] *= float(config.theta_alpha_multiplier)

        normalized_power = power[:, :, total_tensor]
        normalized_power = normalized_power / normalized_power.sum(dim=-1, keepdim=True).clamp_min(eps)
        spectral_entropy = -(normalized_power * torch.log(normalized_power + eps)).sum(dim=-1)

        theta = relative_band_power[:, :, 1]
        alpha = relative_band_power[:, :, 2]
        beta = (relative_band_power[:, :, 3] + relative_band_power[:, :, 4]).clamp_min(eps)
        ratio_theta_beta = (theta / beta).unsqueeze(-1)
        ratio_alpha_beta = (alpha / beta).unsqueeze(-1)

        time_mean = x.mean(dim=-1)
        time_std = x.std(dim=-1, unbiased=False)
        time_rms = torch.sqrt((x**2).mean(dim=-1) + eps)
        time_peak_to_peak = x.amax(dim=-1) - x.amin(dim=-1)

        absolute_flat = absolute_band_power.reshape(absolute_band_power.shape[0], -1)
        relative_flat = relative_band_power.reshape(relative_band_power.shape[0], -1)
        entropy_flat = spectral_entropy.reshape(spectral_entropy.shape[0], -1)
        ratio_flat = torch.cat([ratio_theta_beta, ratio_alpha_beta], dim=-1).reshape(
            spectral_entropy.shape[0], -1
        )
        time_flat = torch.cat([time_mean, time_std, time_rms, time_peak_to_peak], dim=-1)

        line_length = torch.abs(x[..., 1:] - x[..., :-1]).mean(dim=-1)
        alpha_channels = relative_band_power[:, :, 2]
        half_channels = alpha_channels.shape[1] // 2
        alpha_asymmetry = (
            alpha_channels[:, :half_channels].mean(dim=1)
            - alpha_channels[:, half_channels:].mean(dim=1)
        ).unsqueeze(-1)
        delta_theta = (relative_band_power[:, :, 0] + relative_band_power[:, :, 1]).clamp_min(eps)
        alpha_beta = (
            relative_band_power[:, :, 2]
            + relative_band_power[:, :, 3]
            + relative_band_power[:, :, 4]
        ).clamp_min(eps)
        slow_to_fast = (delta_theta / alpha_beta).unsqueeze(-1).reshape(x.shape[0], -1)
        physiological_extra = torch.cat([line_length, alpha_asymmetry, slow_to_fast], dim=-1)

        first_difference = x[..., 1:] - x[..., :-1]
        second_difference = first_difference[..., 1:] - first_difference[..., :-1]
        variance_zero = x.var(dim=-1, unbiased=False).clamp_min(eps)
        variance_one = first_difference.var(dim=-1, unbiased=False).clamp_min(eps)
        variance_two = second_difference.var(dim=-1, unbiased=False).clamp_min(eps)
        mobility = torch.sqrt(variance_one / variance_zero)
        complexity = torch.sqrt(variance_two / variance_one) / mobility.clamp_min(eps)
        activity = torch.log(variance_zero + eps)
        hjorth_flat = torch.cat([activity, mobility, complexity], dim=-1).reshape(x.shape[0], -1)

        gamma_channels = relative_band_power[:, :, -1].clamp_min(eps)
        beta_channels = (
            relative_band_power[:, :, 3] + relative_band_power[:, :, 4]
        ).clamp_min(eps)
        alpha_channels_safe = relative_band_power[:, :, 2].clamp_min(eps)
        theta_channels_safe = relative_band_power[:, :, 1].clamp_min(eps)
        log_channel_ratios = torch.stack(
            [
                torch.log((alpha_channels_safe / beta_channels).mean(dim=1)),
                torch.log((gamma_channels / alpha_channels_safe).mean(dim=1)),
                torch.log((theta_channels_safe / alpha_channels_safe).mean(dim=1)),
            ],
            dim=-1,
        )

        relative_mean = relative_band_power.mean(dim=1)
        relative_std = relative_band_power.std(dim=1, unbiased=False)
        absolute_mean = absolute_band_power.mean(dim=1)
        entropy_mean = spectral_entropy.mean(dim=1, keepdim=True)
        entropy_std = spectral_entropy.std(dim=1, unbiased=False, keepdim=True)
        spatial_flat = torch.cat(
            [relative_mean, relative_std, absolute_mean, entropy_mean, entropy_std],
            dim=-1,
        )

        frontal_theta = relative_band_power[:, :half_channels, 1].mean(dim=1, keepdim=True)
        frontal_beta = (
            relative_band_power[:, :half_channels, 3]
            + relative_band_power[:, :half_channels, 4]
        ).mean(dim=1, keepdim=True).clamp_min(eps)
        frontal_alpha = relative_band_power[:, :half_channels, 2].mean(
            dim=1, keepdim=True
        ).clamp_min(eps)
        workload_features = torch.cat(
            [
                torch.log((frontal_theta / frontal_beta).clamp_min(eps)),
                torch.log(frontal_alpha),
            ],
            dim=-1,
        )

        differential_entropy = []
        for mask in band_masks:
            mask_tensor = torch.from_numpy(mask.astype(np.bool_)).to(device)
            band_signal = torch.fft.irfft(
                spectrum * mask_tensor[None, None, :].float(),
                n=n_fft,
                dim=-1,
            )[..., :win_samples]
            band_variance = band_signal.var(dim=-1, unbiased=False).clamp_min(eps)
            differential_entropy.append(
                0.5 * torch.log(2.0 * math.pi * math.e * band_variance)
            )
        differential_entropy_all = torch.stack(differential_entropy, dim=-1)

        features = torch.cat(
            [
                absolute_flat,
                relative_flat,
                entropy_flat,
                ratio_flat,
                time_flat,
                spatial_flat,
                hjorth_flat,
                physiological_extra,
                log_channel_ratios,
                workload_features,
                differential_entropy_all.reshape(x.shape[0], -1),
                differential_entropy_all.mean(dim=1),
            ],
            dim=-1,
        )
        if features.shape[1] != 272:
            raise AssertionError(f"Expected 272 features, got {features.shape[1]}")

        covariance = torch.matmul(x, x.transpose(1, 2)) / float(x.shape[-1])
        identity = torch.eye(8, device=device, dtype=x.dtype).unsqueeze(0)
        covariance = covariance + 1e-4 * identity

        feature_parts.append(features.detach().cpu().numpy().astype(np.float32))
        covariance_parts.append(covariance.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feature_parts), np.concatenate(covariance_parts)
