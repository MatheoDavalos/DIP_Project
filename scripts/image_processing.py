"""Funções puras de filtragem, segmentação e descritores de folhas."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def filtro_mediana(image, kernel=3):
    return cv2.medianBlur(image, kernel)


def filtro_bilateral(image, diameter=11, sigma_color=30, sigma_space=5):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def filtro_gaussiano(image, kernel=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def filtro_nlm(image, h, h_color, template_window=7, search_window=21):
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=h, hColor=h_color,
        templateWindowSize=template_window, searchWindowSize=search_window)


def filtro_bm3d(image, sigma_psd, profile="np", stage="ALL_STAGES"):
    from bm3d import bm3d, BM3DStages
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    result = bm3d(rgb, sigma_psd=sigma_psd, profile=profile,
                  stage_arg=getattr(BM3DStages, stage))
    result = (np.clip(result, 0, 1) * 255 + .5).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


FILTROS_DISPONIVEIS = {
    "median": filtro_mediana,
    "bilateral": filtro_bilateral,
    "gaussian": filtro_gaussiano,
    "nlm": filtro_nlm,
    "bm3d": filtro_bm3d,
}


def aplicar_filtros(imagens, metodos, parametros):
    resultados = {}
    for metodo in metodos:
        aplicar = FILTROS_DISPONIVEIS[metodo]
        resultados[metodo] = {
            classe: aplicar(imagem, **parametros[metodo])
            for classe, imagem in imagens.items()
        }
    return resultados


def sobel(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def canny(image, low=50, high=100):
    return cv2.Canny(image, low, high, apertureSize=3, L2gradient=True) > 0


def validate_gray(image):
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty grayscale uint8 image.")


def crop_roi(image, box):
    """box = (x, y, width, height), with no resizing or background filling."""
    if len(box) != 4 or not all(isinstance(v, (int, np.integer)) for v in box):
        raise ValueError("ROI coordinates must be four integers.")
    x, y, width, height = box
    rows, cols = image.shape[:2]
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > cols or y + height > rows:
        raise ValueError("ROI must lie entirely inside the image.")
    return image[y:y + height, x:x + width].copy()


def glcm(image, levels=32, distances=(1, 2, 4), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    validate_gray(image)
    if not isinstance(levels, int) or not 2 <= levels <= 256:
        raise ValueError("levels must be an integer in [2, 256].")
    if not distances or any(int(d) != d or d < 1 or d >= min(image.shape) for d in distances):
        raise ValueError("Distances must be positive integers smaller than the ROI.")
    if not angles or not np.isfinite(angles).all():
        raise ValueError("Provide finite angles.")
    quantized = (image.astype(np.uint16) * levels // 256).astype(np.uint8)
    return graycomatrix(quantized, distances, angles, levels=levels, symmetric=True, normed=True)


def glcm_properties(matrix):
    names = ("contrast", "dissimilarity", "homogeneity", "energy", "ASM", "correlation")
    return {name: graycoprops(matrix, name) for name in names}


def validate_mask(image, mask):
    if mask.dtype != bool or mask.shape != image.shape[:2] or not mask.any():
        raise ValueError("Expected a nonempty boolean mask aligned with the image.")


def segment_leaf(image, method):
    """Retorna uma máscara da folha; regiões inválidas são falhas explícitas."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR uint8 image.")
    if method == "otsu":
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "kmeans":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32)
        sample = lab[::4, ::4].reshape(-1, 2)
        cv2.setRNGSeed(42)
        _, _, centers = cv2.kmeans(sample, 3, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .2),
            3, cv2.KMEANS_PP_CENTERS)
        labels = ((lab[:, :, None] - centers) ** 2).sum(axis=3).argmin(axis=2)
        border = np.concatenate((labels[0], labels[-1], labels[1:-1, 0], labels[1:-1, -1]))
        background = np.bincount(border, minlength=3).argmax()
        mask = (labels != background).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown segmentation: {method}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = binary_fill_holes(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        raise ValueError("Empty leaf mask.")
    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    mask = labels == largest
    if not .01 <= mask.mean() <= .98:
        raise ValueError("Leaf mask occupies less than 1% or more than 98% of the image.")
    return mask


def masked_glcm(image, mask, levels=32, distances=(1, 2, 4),
                angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """A máscara seleciona pares; o fundo nunca é convertido em intensidade zero."""
    validate_gray(image)
    validate_mask(image, mask)
    if not isinstance(levels, int) or not 2 <= levels <= 256:
        raise ValueError("levels must be an integer in [2, 256].")
    if not distances or any(int(d) != d or d < 1 or d >= min(image.shape) for d in distances):
        raise ValueError("Invalid GLCM distances.")
    if not angles or not np.isfinite(angles).all():
        raise ValueError("Provide finite angles.")
    quantized = (image.astype(np.uint16) * levels // 256).astype(np.int32)
    height, width = image.shape
    matrix = np.zeros((levels, levels, len(distances), len(angles)), dtype=float)
    for d_index, distance in enumerate(distances):
        for a_index, angle in enumerate(angles):
            dy, dx = int(np.rint(distance * np.sin(angle))), int(np.rint(distance * np.cos(angle)))
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            valid = mask[y0:y1, x0:x1] & mask[y0+dy:y1+dy, x0+dx:x1+dx]
            if not valid.any():
                raise ValueError("No valid pixel pairs for this GLCM offset.")
            source = quantized[y0:y1, x0:x1][valid]
            target = quantized[y0+dy:y1+dy, x0+dx:x1+dx][valid]
            counts = np.bincount(source * levels + target, minlength=levels**2).reshape(levels, levels)
            counts = counts + counts.T
            matrix[:, :, d_index, a_index] = counts / counts.sum()
    return matrix


def shape_features(mask):
    validate_mask(mask, mask)
    regions = regionprops(mask.astype(np.uint8))
    region = regions[0]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or region.area_convex == 0:
        raise ValueError("Degenerate leaf contour.")
    return {
        "shape_area": float(mask.mean()),
        "shape_perimeter": perimeter / np.sqrt(mask.size),
        "shape_circularity": 4 * np.pi * cv2.contourArea(contour) / perimeter**2,
        "shape_eccentricity": region.eccentricity,
        "shape_solidity": region.solidity,
        "shape_convexity": cv2.arcLength(cv2.convexHull(contour), True) / perimeter,
    }


def color_features(image, mask, space):
    validate_mask(image, mask)
    if space not in ("rgb", "hsv"):
        raise ValueError("Color space must be rgb or hsv.")
    code = cv2.COLOR_BGR2RGB if space == "rgb" else cv2.COLOR_BGR2HSV
    channels = cv2.cvtColor(image, code)[mask].astype(float)
    channels /= np.array([180, 256, 256]) if space == "hsv" else 256
    features = {}
    for channel in range(3):
        values = channels[:, channel]
        histogram, _ = np.histogram(values, bins=16, range=(0, 1))
        for bin_index, value in enumerate(histogram / len(values)):
            features[f"{space}_{channel}_hist_{bin_index}"] = value
        if space == "hsv" and channel == 0:
            sin_mean = np.sin(2 * np.pi * values).mean()
            cos_mean = np.cos(2 * np.pi * values).mean()
            moments = (sin_mean, cos_mean, 1 - np.hypot(sin_mean, cos_mean))
        else:
            moments = (values.mean(), values.std(), np.cbrt(((values-values.mean())**3).mean()))
        for index, value in enumerate(moments):
            features[f"{space}_{channel}_moment_{index}"] = value
    return features


def texture_features(gray, mask):
    matrix = masked_glcm(gray, mask)
    features = {f"texture_{name}": float(values.mean())
                for name, values in glcm_properties(matrix).items()}
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if not interior.any():
        raise ValueError("No complete LBP neighborhoods inside the leaf.")
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    histogram = np.bincount(lbp[interior].astype(int), minlength=10)
    features.update({f"texture_lbp_{i}": value for i, value in enumerate(histogram/histogram.sum())})
    return features


def leaf_features(image, mask):
    return {**shape_features(mask), **color_features(image, mask, "rgb"),
            **color_features(image, mask, "hsv"), **texture_features(grayscale(image), mask)}
