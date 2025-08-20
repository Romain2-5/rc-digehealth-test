from scipy import signal
from typing import Union
import numpy as np


class Filter:
    """Abstract base class for signal filters.

    Attributes:
        cutoff (Union[float, list[float]]): Cutoff frequency/frequencies in Hz.
        fs (float): Sampling frequency in Hz.
        btype (str): Filter type (e.g., "lowpass", "highpass", "bandpass", "bandstop").
        axis (int): Axis along which to apply the filter.
    """

    def __init__(self, cutoff: Union[float, list[float]], fs: float, btype: str, axis: int = 0) -> None:
        """
        Args:
            cutoff (Union[float, list[float]]): Cutoff frequency/frequencies in Hz.
             Give a float for unique cutoff, or list of floats if bandpass is used.
            fs (float): Sampling frequency in Hz.
            btype (str): Filter type (e.g., "lowpass", "highpass", "bandpass", "bandstop").
            axis (int, optional): Axis along which to apply the filter. Defaults to 0.
        """

        self.cutoff = cutoff
        self.fs = fs
        self.btype = btype
        self.axis = axis

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to input data.

        Args:
            data (np.ndarray): Input signal array.

        Returns:
            np.ndarray: Filtered signal array.
        """

        raise NotImplementedError()


class ButterFilter(Filter):
    """Butterworth filter implementation using second-order sections (SOS).

    This filter is applied with zero-phase filtering (`sosfiltfilt`) to avoid phase distortion.
    """

    def __init__(
        self,
        cutoff: Union[float, list[float]],
        fs: float,
        btype: str,
        axis: int = 0,
        order: int = 2,
    ) -> None:
        """
        Args:
            cutoff (Union[float, list[float]]): Cutoff frequency/frequencies in Hz.
            fs (float): Sampling frequency in Hz.
            btype (str): Filter type (e.g., "lowpass", "highpass", "bandpass", "bandstop").
            axis (int, optional): Axis along which to apply the filter. Defaults to 0.
            order (int, optional): Filter order. Defaults to 2.
        """

        super().__init__(cutoff, fs, btype, axis)
        self.sos = signal.butter(N=order, Wn=self.cutoff, fs=fs, btype=btype, output="sos")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the Butterworth filter to input data.

        Args:
            data (np.ndarray): Input signal array.

        Returns:
            np.ndarray: Filtered signal array.
        """
        sig = signal.sosfiltfilt(sos=self.sos, x=data, axis=self.axis)

        return sig
