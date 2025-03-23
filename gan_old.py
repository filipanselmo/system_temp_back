from torchvision import transforms
import torch
from torch import nn

class GANGenerator(nn.Module):
    def __init__(self):
        super(GANGenerator, self).__init__()
        # Пример простой архитектуры
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # ... дополнительные слои ...
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def augment_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)
    # Применение предобученного GAN
    gan = GANGenerator()
    gan.load_state_dict(torch.load('gan_weights.pth'))
    with torch.no_grad():
        augmented = gan(img_tensor.unsqueeze(0))
    return transforms.ToPILImage()(augmented.squeeze(0))