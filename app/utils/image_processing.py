import cv2
import numpy as np
import os
import easyocr

def save_image(image: np.ndarray, step_name: str) -> None:
    output_dir: str = 'processed_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, f'{step_name}.jpg'), image)


def ensure_color(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Formato de imagen no soportado")


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return image.squeeze()
    else:
        raise ValueError("Formato de imagen no soportado")

def rotate_if_vertical(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        print("La imagen es vertical, rotando 90 grados en sentido antihorario...")
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

# def super_resolution(image: np.ndarray) -> np.ndarray:
#     # initialize super resolution object
#     sr = dnn_superres.DnnSuperResImpl_create()
#
#     # read the model
#     path = 'EDSR_x4.pb'
#     sr.readModel(path)
#
#     # set the model and scale
#     sr.setModel('edsr', 2)
#
#     # if you have cuda support
#     sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#
#     # upsample the image
#     upscaled = sr.upsample(image)
#
#     return upscaled

def resize_image(image: np.ndarray, scale_factor: float = 4.0) -> np.ndarray:
    """
    Redimensiona la imagen por un factor de escala.

    :param image: Imagen a redimensionar
    :param scale_factor: Factor de escala (2.0 significa duplicar el tamaño)
    :return: Imagen redimensionada
    """
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

def align_image(image: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            return rotated

    return image


def remove_noise(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        # Imagen en escala de grises
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Imagen en color
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        raise ValueError("Formato de imagen no soportado para eliminación de ruido")

def improve_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def detect_edges(image: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(image)
    return cv2.Canny(gray, 50, 150, apertureSize=3)


def apply_threshold(image: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(image)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def block_otsu(image: np.ndarray, block_size: int = 64) -> np.ndarray:
    gray = ensure_grayscale(image)
    output = np.zeros_like(gray)
    for y in range(0, gray.shape[0], block_size):
        for x in range(0, gray.shape[1], block_size):
            block = gray[y:y + block_size, x:x + block_size]
            threshold, _ = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            output[y:y + block_size, x:x + block_size] = cv2.threshold(block, threshold, 255, cv2.THRESH_BINARY)[1]
    return output


def preprocess_image(image: np.ndarray, reader: easyocr.Reader) -> np.ndarray:
    print("Iniciando procesamiento de imagen")

    save_image(image, 'original')

    print("Verificando orientación de la imagen...")
    image = rotate_if_vertical(image)
    save_image(image, 'rotated_if_needed')

    print("Alineando imagen...")
    image = align_image(image)
    save_image(image, 'aligned')

    print("Redimensionando imagen...")
    image = resize_image(image)
    save_image(image, 'resized')

    # print("Aplicando super-resolución...")
    # image = super_resolution(image)
    # save_image(image, 'super_resolution')

    # print("Eliminando ruido...")
    # image = remove_noise(image)
    # save_image(image, 'denoised')

    # print("Mejorando contraste...")
    # image = ensure_color(image)  # Aseguramos que la imagen esté en color
    # image = improve_contrast(image)
    # save_image(image, 'contrast_improved')

    # print("Detectando bordes...")
    # edges = detect_edges(image)
    # save_image(edges, 'edges_detected')
    #
    # print("Convirtiendo a escala de grises...")
    # gray = ensure_grayscale(image)
    # save_image(gray, 'grayscale')
    #
    # print("Aplicando umbral...")
    # threshold = apply_threshold(gray)
    # save_image(threshold, 'threshold')
    #
    # print("Aplicando umbral de Otsu por bloques...")
    # block_threshold = block_otsu(threshold)
    # save_image(block_threshold, 'block_otsu')

    return image