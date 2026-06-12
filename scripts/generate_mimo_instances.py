#!/usr/bin/env python3
"""
Generate SATField/MIMOPotts-compatible MIMO detection instances.

The output .npz files match the loader contract used by
MIMOPotts.load_mimo_potts_instance:

  nt, nr, modulation, channel_model, ebnodb, no,
  H, y, x, Haug, yaug, xaug

The generator intentionally depends only on NumPy. It covers reproducible
Rayleigh, correlated Rayleigh, Rician, and controlled ill-conditioned channels.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SUPPORTED_MODULATIONS = ("QPSK", "16QAM", "64QAM", "256QAM", "1024QAM")
SUPPORTED_CHANNEL_MODELS = ("Rayleigh", "CorrelatedRayleigh", "Rician", "IllConditionedRayleigh")
EBNODB_GRID_0_TO_30_DB = tuple(float(x) for x in np.linspace(0.0, 30.0, 30))
SNR_GRID_QPSK = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
SNR_GRID_16QAM = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0)
SNR_GRID_64QAM = (6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0)
SNR_GRID_256QAM = (10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0)
SNR_GRID_1024QAM = (16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0)
SNR_GRID_SCALING = (6.0, 12.0, 18.0, 24.0, 30.0)
SNR_GRID_CERT_QPSK = (2.0, 4.0, 6.0, 8.0, 10.0)
SNR_GRID_CERT_16QAM = (6.0, 9.0, 12.0, 15.0, 18.0, 21.0)
SNR_GRID_CERT_64QAM = (12.0, 15.0, 18.0, 21.0, 24.0, 27.0)
SNR_GRID_LOWBER_16QAM = (15.0, 18.0, 21.0)
SNR_GRID_LOWBER_64QAM = (21.0, 24.0, 27.0)
SNR_GRID_LOWBER_256QAM = (25.0, 28.0, 31.0)


@dataclass(frozen=True)
class Scenario:
    name: str
    nt: int
    nr: int
    modulation: str
    ebnodb: tuple[float, ...]
    instances: int
    seed: int
    channel_model: str = "Rayleigh"
    correlation_rho: float = 0.0
    rician_k: float = 0.0
    condition_number: float = 1.0


SMOKE_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="A_dbg_2x2_qpsk_rayleigh",
        nt=2,
        nr=2,
        modulation="QPSK",
        ebnodb=EBNODB_GRID_0_TO_30_DB,
        instances=16,
        seed=1,
    ),
)


CORE_PAPER_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("core_4x4_16qam_rayleigh", 4, 4, "16QAM", SNR_GRID_16QAM, 2000, 101),
    Scenario("core_8x8_16qam_rayleigh", 8, 8, "16QAM", SNR_GRID_16QAM, 1500, 102),
    Scenario("core_8x8_64qam_rayleigh", 8, 8, "64QAM", SNR_GRID_64QAM, 1500, 103),
    Scenario("core_16x16_16qam_rayleigh", 16, 16, "16QAM", SNR_GRID_16QAM, 1000, 104),
    Scenario("core_16x16_64qam_rayleigh", 16, 16, "64QAM", SNR_GRID_64QAM, 1000, 105),
    Scenario("core_16x32_16qam_rayleigh", 16, 32, "16QAM", SNR_GRID_16QAM, 800, 106),
    Scenario("core_16x32_256qam_rayleigh", 16, 32, "256QAM", SNR_GRID_256QAM, 800, 107),
)


CERTIFIED_ML_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("certml_2x2_qpsk_rayleigh", 2, 2, "QPSK", SNR_GRID_CERT_QPSK, 5000, 151),
    Scenario("certml_4x4_16qam_rayleigh", 4, 4, "16QAM", SNR_GRID_CERT_16QAM, 2000, 152),
    Scenario("certml_4x4_64qam_rayleigh", 4, 4, "64QAM", SNR_GRID_CERT_64QAM, 1000, 153),
    Scenario("certml_8x8_16qam_rayleigh", 8, 8, "16QAM", SNR_GRID_CERT_16QAM, 500, 154),
    Scenario(
        "certml_4x4_16qam_corr07",
        4,
        4,
        "16QAM",
        SNR_GRID_CERT_16QAM,
        1000,
        155,
        channel_model="CorrelatedRayleigh",
        correlation_rho=0.7,
    ),
)


LOW_BER_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("lowber_4x4_16qam_rayleigh", 4, 4, "16QAM", SNR_GRID_LOWBER_16QAM, 20000, 251),
    Scenario("lowber_8x8_16qam_rayleigh", 8, 8, "16QAM", SNR_GRID_LOWBER_16QAM, 12000, 252),
    Scenario("lowber_8x8_64qam_rayleigh", 8, 8, "64QAM", SNR_GRID_LOWBER_64QAM, 10000, 253),
    Scenario("lowber_16x16_64qam_rayleigh", 16, 16, "64QAM", SNR_GRID_LOWBER_64QAM, 5000, 254),
    Scenario("lowber_16x32_16qam_rayleigh", 16, 32, "16QAM", SNR_GRID_LOWBER_16QAM, 4000, 255),
    Scenario("lowber_16x32_256qam_rayleigh", 16, 32, "256QAM", SNR_GRID_LOWBER_256QAM, 3000, 256),
    Scenario(
        "lowber_16x16_64qam_corr07",
        16,
        16,
        "64QAM",
        SNR_GRID_LOWBER_64QAM,
        4000,
        257,
        channel_model="CorrelatedRayleigh",
        correlation_rho=0.7,
    ),
)


ROBUSTNESS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "robust_16x16_64qam_corr03",
        16,
        16,
        "64QAM",
        SNR_GRID_64QAM,
        750,
        201,
        channel_model="CorrelatedRayleigh",
        correlation_rho=0.3,
    ),
    Scenario(
        "robust_16x16_64qam_corr07",
        16,
        16,
        "64QAM",
        SNR_GRID_64QAM,
        750,
        202,
        channel_model="CorrelatedRayleigh",
        correlation_rho=0.7,
    ),
    Scenario(
        "robust_16x16_64qam_corr09",
        16,
        16,
        "64QAM",
        SNR_GRID_64QAM,
        750,
        203,
        channel_model="CorrelatedRayleigh",
        correlation_rho=0.9,
    ),
    Scenario(
        "robust_16x32_256qam_ricianK10",
        16,
        32,
        "256QAM",
        SNR_GRID_256QAM,
        500,
        204,
        channel_model="Rician",
        rician_k=10.0,
    ),
    Scenario(
        "robust_8x8_64qam_illcond100",
        8,
        8,
        "64QAM",
        SNR_GRID_64QAM,
        1000,
        205,
        channel_model="IllConditionedRayleigh",
        condition_number=100.0,
    ),
    Scenario("robust_overloaded_16Rx_24Users_qpsk", 24, 16, "QPSK", SNR_GRID_QPSK, 1000, 206),
    Scenario("robust_overloaded_24Rx_32Users_16qam", 32, 24, "16QAM", SNR_GRID_16QAM, 750, 207),
)


SCALING_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("scale_64Rx_128Users_256qam_rayleigh", 128, 64, "256QAM", SNR_GRID_256QAM, 64, 301),
    Scenario("scale_128Rx_32Users_64qam_rayleigh", 32, 128, "64QAM", SNR_GRID_64QAM, 250, 302),
    Scenario("scale_256Rx_64Users_256qam_rayleigh", 64, 256, "256QAM", SNR_GRID_SCALING, 48, 303),
    Scenario("scale_512Rx_128Users_64qam_rayleigh", 128, 512, "64QAM", SNR_GRID_SCALING, 24, 304),
    Scenario("scale_overloaded_128Rx_256Users_qpsk", 256, 128, "QPSK", SNR_GRID_QPSK, 32, 305),
    Scenario(
        "scale_256Rx_32Users_1024qam_ricianK10",
        32,
        256,
        "1024QAM",
        SNR_GRID_1024QAM,
        32,
        306,
        channel_model="Rician",
        rician_k=10.0,
    ),
)


LEGACY_SATFIELD_RAYLEIGH_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("A_dbg_2x2_qpsk_rayleigh", 2, 2, "QPSK", EBNODB_GRID_0_TO_30_DB, 2000, 1),
    Scenario("B_dbg_2x2_16qam_rayleigh", 2, 2, "16QAM", EBNODB_GRID_0_TO_30_DB, 2000, 2),
    Scenario("C_mid_4x4_64qam_rayleigh", 4, 4, "64QAM", EBNODB_GRID_0_TO_30_DB, 2000, 3),
    Scenario("D_mid_8x8_16qam_rayleigh", 8, 8, "16QAM", EBNODB_GRID_0_TO_30_DB, 1000, 4),
    Scenario("E_nr_like_16x32_64qam_rayleigh", 16, 32, "64QAM", EBNODB_GRID_0_TO_30_DB, 500, 5),
    Scenario("F_high_order_8x16_1024qam_rayleigh", 8, 16, "1024QAM", EBNODB_GRID_0_TO_30_DB, 500, 6),
)


LEGACY_FUTURE_6G_RAYLEIGH_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("G_massive_ul_64Rx_128Users_256qam_rayleigh", 128, 64, "256QAM", EBNODB_GRID_0_TO_30_DB, 50, 7),
    Scenario("H_mumimo_ul_128Rx_32Users_64qam_rayleigh", 32, 128, "64QAM", EBNODB_GRID_0_TO_30_DB, 250, 8),
    Scenario("I_xl_mimo_ul_256Rx_64Users_256qam_rayleigh", 64, 256, "256QAM", EBNODB_GRID_0_TO_30_DB, 32, 9),
    Scenario("J_cellfree_ul_512Rx_128Users_64qam_rayleigh", 128, 512, "64QAM", EBNODB_GRID_0_TO_30_DB, 16, 10),
    Scenario("K_overloaded_iot_128Rx_256Users_qpsk_rayleigh", 256, 128, "QPSK", EBNODB_GRID_0_TO_30_DB, 24, 11),
    Scenario("L_thz_hotspot_256Rx_32Users_1024qam_rayleigh", 32, 256, "1024QAM", EBNODB_GRID_0_TO_30_DB, 32, 12),
)


PRESETS: dict[str, tuple[Scenario, ...]] = {
    "smoke": SMOKE_SCENARIOS,
    "core-paper": CORE_PAPER_SCENARIOS,
    "certified-ml": CERTIFIED_ML_SCENARIOS,
    "robustness": ROBUSTNESS_SCENARIOS,
    "low-ber": LOW_BER_SCENARIOS,
    "scaling": SCALING_SCENARIOS,
    "all": CORE_PAPER_SCENARIOS + CERTIFIED_ML_SCENARIOS + ROBUSTNESS_SCENARIOS + LOW_BER_SCENARIOS + SCALING_SCENARIOS,
    "legacy-satfield-rayleigh": LEGACY_SATFIELD_RAYLEIGH_SCENARIOS,
    "legacy-future-6g-rayleigh": LEGACY_FUTURE_6G_RAYLEIGH_SCENARIOS,
    "legacy-all": LEGACY_SATFIELD_RAYLEIGH_SCENARIOS + LEGACY_FUTURE_6G_RAYLEIGH_SCENARIOS,
    "satfield-rayleigh": LEGACY_SATFIELD_RAYLEIGH_SCENARIOS,
    "future-6g-rayleigh": LEGACY_FUTURE_6G_RAYLEIGH_SCENARIOS,
}


def parse_ebnodb(values: str) -> tuple[float, ...]:
    out = tuple(float(v.strip()) for v in values.split(",") if v.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one Eb/N0 value")
    return out


def bits_per_symbol(modulation: str) -> int:
    mod = modulation.upper()
    if mod == "QPSK":
        return 2
    if mod.endswith("QAM"):
        return int(round(math.log2(int(mod[:-3]))))
    raise ValueError(f"unsupported modulation: {modulation}")


def pam_levels(modulation: str) -> np.ndarray:
    mod = modulation.upper()
    if mod == "QPSK":
        alpha = 1.0 / math.sqrt(2.0)
        return np.array([-alpha, alpha], dtype=np.float32)

    if mod.endswith("QAM"):
        order = int(mod[:-3])
        bps = bits_per_symbol(mod)
        if bps % 2 != 0:
            raise ValueError(f"expected square QAM, got {mod}")
        nlevels = 1 << (bps // 2)
        gamma = math.sqrt(3.0 / (2.0 * (order - 1.0)))
        return np.array([gamma * (2.0 * u - (nlevels - 1.0)) for u in range(nlevels)], dtype=np.float32)

    raise ValueError(f"unsupported modulation: {modulation}")


def random_symbols(rng: np.random.Generator, n_instances: int, nt: int, modulation: str) -> np.ndarray:
    levels = pam_levels(modulation)
    real_idx = rng.integers(0, len(levels), size=(n_instances, nt))
    imag_idx = rng.integers(0, len(levels), size=(n_instances, nt))
    return (levels[real_idx] + 1j * levels[imag_idx]).astype(np.complex64)


def channel_scale(normalization: str, nr: int, nt: int) -> float:
    if normalization == "none":
        return 1.0
    if normalization == "by-nr":
        return 1.0 / math.sqrt(float(nr))
    if normalization == "by-nt":
        return 1.0 / math.sqrt(float(nt))
    raise ValueError(f"unsupported channel normalization: {normalization}")


def iid_rayleigh_channel(
    rng: np.random.Generator,
    n_instances: int,
    nr: int,
    nt: int,
    normalization: str,
) -> np.ndarray:
    scale = channel_scale(normalization, nr, nt) / math.sqrt(2.0)
    return (scale * (rng.standard_normal((n_instances, nr, nt)) + 1j * rng.standard_normal((n_instances, nr, nt)))).astype(np.complex64)


def exp_corr_sqrt(n: int, rho: float) -> np.ndarray:
    idx = np.arange(n)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.T


def correlated_rayleigh_channel(
    rng: np.random.Generator,
    n_instances: int,
    nr: int,
    nt: int,
    normalization: str,
    rho: float,
) -> np.ndarray:
    if not 0.0 <= rho < 1.0:
        raise ValueError("correlation_rho must be in [0, 1)")
    W = iid_rayleigh_channel(rng, n_instances, nr, nt, normalization)
    Rr = exp_corr_sqrt(nr, rho)
    Rt = exp_corr_sqrt(nt, rho)
    return np.einsum("ab,ibc,cd->iad", Rr, W, Rt).astype(np.complex64)


def rician_channel(
    rng: np.random.Generator,
    n_instances: int,
    nr: int,
    nt: int,
    normalization: str,
    k_factor: float,
) -> np.ndarray:
    if k_factor < 0.0:
        raise ValueError("rician_k must be non-negative")
    if k_factor == 0.0:
        return iid_rayleigh_channel(rng, n_instances, nr, nt, normalization)

    scattered = iid_rayleigh_channel(rng, n_instances, nr, nt, normalization)
    scale = channel_scale(normalization, nr, nt)
    rx_phase = rng.uniform(0.0, 2.0 * math.pi, size=(n_instances, nr, 1))
    tx_phase = rng.uniform(0.0, 2.0 * math.pi, size=(n_instances, 1, nt))
    los = scale * np.exp(1j * (rx_phase + tx_phase))
    return (math.sqrt(k_factor / (k_factor + 1.0)) * los + math.sqrt(1.0 / (k_factor + 1.0)) * scattered).astype(np.complex64)


def ill_conditioned_rayleigh_channel(
    rng: np.random.Generator,
    n_instances: int,
    nr: int,
    nt: int,
    normalization: str,
    condition_number: float,
) -> np.ndarray:
    if condition_number < 1.0:
        raise ValueError("condition_number must be >= 1")

    rank = min(nr, nt)
    singular_values = np.geomspace(1.0, 1.0 / condition_number, num=rank)
    target_frobenius = math.sqrt(float(nr * nt)) * channel_scale(normalization, nr, nt)
    out = np.empty((n_instances, nr, nt), dtype=np.complex64)

    for i in range(n_instances):
        u_seed = (rng.standard_normal((nr, rank)) + 1j * rng.standard_normal((nr, rank))) / math.sqrt(2.0)
        v_seed = (rng.standard_normal((nt, rank)) + 1j * rng.standard_normal((nt, rank))) / math.sqrt(2.0)
        U, _ = np.linalg.qr(u_seed)
        V, _ = np.linalg.qr(v_seed)
        H_i = (U * singular_values) @ V.conj().T
        H_i *= target_frobenius / np.linalg.norm(H_i)
        out[i] = H_i.astype(np.complex64)

    return out


def generate_channel(
    rng: np.random.Generator,
    scenario: Scenario,
    normalization: str,
) -> np.ndarray:
    if scenario.channel_model == "Rayleigh":
        return iid_rayleigh_channel(rng, scenario.instances, scenario.nr, scenario.nt, normalization)
    if scenario.channel_model == "CorrelatedRayleigh":
        return correlated_rayleigh_channel(
            rng,
            scenario.instances,
            scenario.nr,
            scenario.nt,
            normalization,
            scenario.correlation_rho,
        )
    if scenario.channel_model == "Rician":
        return rician_channel(rng, scenario.instances, scenario.nr, scenario.nt, normalization, scenario.rician_k)
    if scenario.channel_model == "IllConditionedRayleigh":
        return ill_conditioned_rayleigh_channel(
            rng,
            scenario.instances,
            scenario.nr,
            scenario.nt,
            normalization,
            scenario.condition_number,
        )
    raise ValueError(f"unsupported channel model: {scenario.channel_model}")


def complex_to_real_aug(H: np.ndarray, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Hr = np.real(H)
    Hi = np.imag(H)
    top = np.concatenate([Hr, -Hi], axis=-1)
    bot = np.concatenate([Hi, Hr], axis=-1)
    Haug = np.concatenate([top, bot], axis=-2)
    yaug = np.concatenate([np.real(y), np.imag(y)], axis=-1)
    xaug = np.concatenate([np.real(x), np.imag(x)], axis=-1)
    return Haug.astype(np.float32), yaug.astype(np.float32), xaug.astype(np.float32)


def ebnodb_to_no(ebnodb: float, modulation: str) -> float:
    return 1.0 / (bits_per_symbol(modulation) * (10.0 ** (ebnodb / 10.0)))


def validate_scenario(scenario: Scenario) -> None:
    if scenario.modulation.upper() not in SUPPORTED_MODULATIONS:
        raise ValueError(f"unsupported modulation {scenario.modulation}; expected one of {SUPPORTED_MODULATIONS}")
    if scenario.channel_model not in SUPPORTED_CHANNEL_MODELS:
        raise ValueError(f"unsupported channel model {scenario.channel_model}; expected one of {SUPPORTED_CHANNEL_MODELS}")
    if scenario.nt <= 0 or scenario.nr <= 0 or scenario.instances <= 0:
        raise ValueError("nt, nr, and instances must be positive")


def generate_scenario(
    scenario: Scenario,
    out_dir: Path,
    *,
    channel_normalization: str,
    overwrite: bool,
) -> Path:
    validate_scenario(scenario)
    modulation = scenario.modulation.upper()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario.name}.npz"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")

    rng = np.random.default_rng(scenario.seed)
    x = random_symbols(rng, scenario.instances, scenario.nt, modulation)
    n_snr = len(scenario.ebnodb)

    H = np.empty((n_snr, scenario.instances, scenario.nr, scenario.nt), dtype=np.complex64)
    y = np.empty((n_snr, scenario.instances, scenario.nr), dtype=np.complex64)
    x_all = np.broadcast_to(x, (n_snr,) + x.shape).copy()
    no = np.array([ebnodb_to_no(snr, modulation) for snr in scenario.ebnodb], dtype=np.float32)

    for si, no_value in enumerate(no):
        H_si = generate_channel(rng, scenario, channel_normalization)
        clean = np.einsum("ijk,ik->ij", H_si, x)
        noise_scale = math.sqrt(float(no_value) / 2.0)
        noise = noise_scale * (
            rng.standard_normal((scenario.instances, scenario.nr))
            + 1j * rng.standard_normal((scenario.instances, scenario.nr))
        )
        H[si] = H_si
        y[si] = (clean + noise).astype(np.complex64)

    Haug, yaug, xaug = complex_to_real_aug(H, y, x_all)

    np.savez_compressed(
        out_path,
        nt=np.int64(scenario.nt),
        nr=np.int64(scenario.nr),
        modulation=np.array(modulation),
        channel_model=np.array(scenario.channel_model),
        ebnodb=np.array(scenario.ebnodb, dtype=np.float32),
        no=no,
        y=y,
        H=H,
        x=x_all,
        Haug=Haug,
        yaug=yaug,
        xaug=xaug,
        seed=np.int64(scenario.seed),
        generator=np.array("MIMOPottsBenchmarks/scripts/generate_mimo_instances.py"),
        channel_normalization=np.array(channel_normalization),
        correlation_rho=np.float32(scenario.correlation_rho),
        rician_k=np.float32(scenario.rician_k),
        condition_number=np.float32(scenario.condition_number),
    )
    return out_path


def custom_scenario(args: argparse.Namespace) -> Scenario:
    return Scenario(
        name=args.name,
        nt=args.nt,
        nr=args.nr,
        modulation=args.modulation.upper(),
        ebnodb=args.ebnodb,
        instances=args.instances,
        seed=args.seed,
        channel_model=args.channel_model,
        correlation_rho=args.correlation_rho,
        rician_k=args.rician_k,
        condition_number=args.condition_number,
    )


def selected_scenarios(args: argparse.Namespace) -> Iterable[Scenario]:
    if args.preset == "custom":
        yield custom_scenario(args)
        return
    yield from PRESETS[args.preset]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MIMO .npz instances compatible with MIMOPotts and the original SATField MIMO data layout.",
    )
    parser.add_argument("--out-dir", default="data/mimo_instances", help="directory for generated .npz files")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS) + ("custom",),
        default="all",
        help="scenario set to generate; all is the current recommended benchmark suite",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument(
        "--channel-normalization",
        choices=("none", "by-nr", "by-nt"),
        default="by-nr",
        help="channel coefficient normalization; use 'none' for legacy CN(0,1) SATField-style channels",
    )

    custom = parser.add_argument_group("custom preset options")
    custom.add_argument("--name", default="custom_2x2_qpsk_rayleigh", help="output file stem for --preset custom")
    custom.add_argument("--nt", type=int, default=2, help="number of transmit streams/users for --preset custom")
    custom.add_argument("--nr", type=int, default=2, help="number of receive antennas for --preset custom")
    custom.add_argument("--modulation", choices=SUPPORTED_MODULATIONS, default="QPSK", help="modulation for --preset custom")
    custom.add_argument("--channel-model", choices=SUPPORTED_CHANNEL_MODELS, default="Rayleigh", help="channel model for --preset custom")
    custom.add_argument("--correlation-rho", type=float, default=0.0, help="exponential Tx/Rx correlation rho for CorrelatedRayleigh")
    custom.add_argument("--rician-k", type=float, default=0.0, help="Rician K-factor for Rician channels")
    custom.add_argument("--condition-number", type=float, default=1.0, help="target condition number for IllConditionedRayleigh")
    custom.add_argument(
        "--ebnodb",
        type=parse_ebnodb,
        default=EBNODB_GRID_0_TO_30_DB,
        help="comma-separated Eb/N0 dB values for --preset custom",
    )
    custom.add_argument("--instances", type=int, default=16, help="number of independent instances for --preset custom")
    custom.add_argument("--seed", type=int, default=1, help="random seed for --preset custom")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    for scenario in selected_scenarios(args):
        out_path = generate_scenario(
            scenario,
            out_dir,
            channel_normalization=args.channel_normalization,
            overwrite=args.overwrite,
        )
        print(
            f"Saved {out_path} "
            f"(nt={scenario.nt}, nr={scenario.nr}, modulation={scenario.modulation}, "
            f"channel={scenario.channel_model}, normalization={args.channel_normalization}, "
            f"snrs={len(scenario.ebnodb)}, instances={scenario.instances})"
        )


if __name__ == "__main__":
    main()
