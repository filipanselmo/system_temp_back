import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import io

# Загружаем предобученную SRGAN модель для супер-резолюции
# Убедитесь, что модель доступна в PyTorch Hub
try:
    # srgan_model = torch.hub.load('ai-forever/Real-ESRGAN', 'ESRGAN', pretrained=True)
    # srgan_model.eval()

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    srgan_model = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=0
    )
except Exception as e:
    raise ImportError(
        f"Ошибка загрузки SRGAN: {str(e)}. Установите зависимости: pip install git+https://github.com/ai-forever/Real-ESRGAN.git")


# def augment_image(image: Image) -> Image:
#     """Нормализация и аугментация изображения через SRGAN"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     srgan_model.to(device)
#
#     # Подготовка изображения
#     preprocess = transforms.Compose([
#         transforms.Resize((512, 512)),  # SRGAN требует кратные 4 размеры
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # Конвертация изображения
#     img_tensor = preprocess(image).unsqueeze(0).to(device)
#
#     # Улучшение качества через GAN
#     with torch.no_grad():
#         upscaled_tensor = srgan_model(img_tensor)
#
#     # Постобработка
#     postprocess = transforms.Compose([
#         transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
#         transforms.ToPILImage()
#     ])
#
#     return postprocess(upscaled_tensor.squeeze(0).cpu())

# def augment_image(image: Image) -> Image:
#     """Улучшение качества изображения"""
#     img = np.array(image)
#     upscaled, _ = srgan_model.enhance(img)
#     return Image.fromarray(upscaled)

def augment_image(image: Image) -> Image:
    """Улучшение качества изображения с сохранением 3 каналов"""
    # Приводим к RGB, если изображение grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img = np.array(image)

    # Убедитесь, что SRGAN работает с 3 каналами
    if img.ndim == 2:  # Если одноканальное
        img = np.stack([img] * 3, axis=-1)  # Превращаем в 3 канала

    upscaled, _ = srgan_model.enhance(img)

    # Создаем непрерывный массив
    upscaled = np.ascontiguousarray(upscaled)

    # Конвертируем обратно в RGB, если результат в grayscale
    result = Image.fromarray(upscaled)
    return result.convert('RGB')  # Принудительно сохраняем 3 канала