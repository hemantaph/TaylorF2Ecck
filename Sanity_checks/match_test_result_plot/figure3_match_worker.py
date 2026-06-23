"""Multiprocessing workers for Figure3.ipynb.

Jupyter notebooks on macOS/Python 3.10 use spawned worker processes by
default. Spawned workers cannot import functions defined only in notebook
cells, so the match calculations live in this module.
"""

import bilby
import numpy as np
from pycbc.filter.matchedfilter import match
from pycbc.types.frequencyseries import FrequencySeries


GAMMA = 0.5772156649015329
PI = np.pi
MTSUN_SI = 4.925491025543576e-06

TIME_OF_EVENT = 1126259642.413
POST_TRIGGER_DURATION = 1.0
WAVEFORM_DURATION = 309.0
ANALYSIS_START = TIME_OF_EVENT + POST_TRIGGER_DURATION - WAVEFORM_DURATION
SAMPLING_FREQUENCY = 2048.0
F_MIN = 20.0
F_MAX = 1024.0

BASE_INJECTION_PARAMETERS = {
    "mass_1": 1.61,
    "mass_2": 1.18,
    "eccentricity": 0.1,
    "a_1": 0.0,
    "a_2": 0.0,
    "tilt_1": 0.0,
    "tilt_2": 0.0,
    "phi_12": 0.0,
    "phi_jl": 0.0,
    "luminosity_distance": 38.87,
    "theta_jn": 0.4,
    "psi": 2.659,
    "phase": 1.3,
    "geocent_time": 1126259642.413,
    "ra": 1.375,
    "dec": -1.2108,
}


def findchirp_chirptime(m1, m2, fmin):
    m = m1 + m2
    eta = m1 * m2 / m / m
    c0T = c2T = c3T = c4T = c5T = c6T = c6LogT = c7T = 0.0

    c7T = PI * (
        14809.0 * eta * eta / 378.0 - 75703.0 * eta / 756.0 - 15419335.0 / 127008.0
    )

    c6T = (
        GAMMA * 6848.0 / 105.0
        - 10052469856691.0 / 23471078400.0
        + PI * PI * 128.0 / 3.0
        + eta * (3147553127.0 / 3048192.0 - PI * PI * 451.0 / 12.0)
        - eta * eta * 15211.0 / 1728.0
        + eta * eta * eta * 25565.0 / 1296.0
        + eta * eta * eta * 25565.0 / 1296.0
        + np.log(4.0) * 6848.0 / 105.0
    )
    c6LogT = 6848.0 / 105.0

    c5T = 13.0 * PI * eta / 3.0 - 7729.0 * PI / 252.0

    c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0)
    c3T = -32.0 * PI / 5.0
    c2T = 743.0 / 252.0 + eta * 11.0 / 3.0
    c0T = 5.0 * m * MTSUN_SI / (256.0 * eta)

    xT = np.power(PI * m * MTSUN_SI * fmin, 1.0 / 3.0)
    x2T = xT * xT
    x3T = xT * x2T
    x4T = x2T * x2T
    x5T = x2T * x3T
    x6T = x3T * x3T
    x7T = x3T * x4T
    x8T = x4T * x4T

    return (
        c0T
        * (
            1
            + c2T * x2T
            + c3T * x3T
            + c4T * x4T
            + c5T * x5T
            + (c6T + c6LogT * np.log(xT)) * x6T
            + c7T * x7T
        )
        / x8T
    )


def _make_waveform_generator(waveform_arguments):
    return bilby.gw.WaveformGenerator(
        duration=WAVEFORM_DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_eccentric_binary_black_hole_no_spins,
        parameters=BASE_INJECTION_PARAMETERS,
        waveform_arguments=dict(waveform_arguments),
    )


def _match_row(i, size, waveform_arguments1, waveform_arguments2, psd_array, masses_for_column):
    waveform_generator1 = _make_waveform_generator(waveform_arguments1)
    waveform_generator2 = _make_waveform_generator(waveform_arguments2)

    match_array_ab = np.zeros(size)
    e0_arr = np.linspace(0.01, 0.1, size)
    e0 = e0_arr[i]

    for j in range(size):
        mass_1_, mass_2_ = masses_for_column(j)
        injection_parameters = dict(
            BASE_INJECTION_PARAMETERS,
            mass_1=mass_1_,
            mass_2=mass_2_,
            eccentricity=e0,
        )

        safety = 1.2
        approx_duration = safety * findchirp_chirptime(mass_1_, mass_2_, F_MIN)
        duration = np.ceil(approx_duration + 2.0)

        if duration < 4:
            duration = 4.0

        polas_a = waveform_generator1.frequency_domain_strain(parameters=injection_parameters)
        polas_b = waveform_generator2.frequency_domain_strain(parameters=injection_parameters)
        arr_a = FrequencySeries(polas_a["plus"], delta_f=1 / duration)
        arr_b = FrequencySeries(polas_b["plus"], delta_f=1 / duration)
        psd_ = FrequencySeries(psd_array, delta_f=1 / duration)
        match_array_ab[j] = match(
            arr_a,
            arr_b,
            psd=psd_,
            low_frequency_cutoff=F_MIN,
            high_frequency_cutoff=F_MAX,
        )[0]

    return i, (1 - match_array_ab) * 100


def find_matchvalues_fixed_m(input_arguments):
    i, size, waveform_arguments1, waveform_arguments2, mass_ratio, psd_array = input_arguments
    mtot_arr = np.linspace(2.0, 6.0, size)

    def masses_for_column(j):
        mtot = mtot_arr[j]
        mass_1 = mtot / (1 + mass_ratio)
        mass_2 = mass_1 * mass_ratio
        return mass_1, mass_2

    return _match_row(
        i,
        size,
        waveform_arguments1,
        waveform_arguments2,
        psd_array,
        masses_for_column,
    )


def find_matchvalues_fixed_eta(input_arguments):
    i, size, waveform_arguments1, waveform_arguments2, mtot, psd_array = input_arguments
    eta = np.linspace(0.1875, 0.25, size)
    sqrt_term = np.sqrt(1 - 4 * eta)
    q_arr = (1 - 2 * eta - sqrt_term) / (2 * eta)

    def masses_for_column(j):
        mass_ratio = q_arr[j]
        mass_1 = mtot / (1 + mass_ratio)
        mass_2 = mass_1 * mass_ratio
        return mass_1, mass_2

    return _match_row(
        i,
        size,
        waveform_arguments1,
        waveform_arguments2,
        psd_array,
        masses_for_column,
    )
