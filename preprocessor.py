import cv2
import numpy as np


class WeatherEnhancer:
    """Preprocesses images degraded by adverse weather (rain, fog, low contrast)."""

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel_enhanced = clahe.apply(l_channel)

        merged = cv2.merge((l_channel_enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def reduce_rain_streaks(image: np.ndarray) -> np.ndarray:
        """Applies an edge-preserving bilateral filter to suppress high-frequency rain streaks."""
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def process(self, image: np.ndarray, apply_derain: bool = True) -> np.ndarray:
        """Full adverse-weather enhancement pipeline."""
        enhanced = self.enhance_contrast(image)
        if apply_derain:
            enhanced = self.reduce_rain_streaks(enhanced)
        return enhanced