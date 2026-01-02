# gradcam_demo.py
# ------------------------------------------------------------
# Grad-CAM visualization for Brain MRI ResNet18
# - best_resnet18.pt 로드
# - TEST 데이터 일부(n개)만 시각화
# ------------------------------------------------------------
import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =====================
# Config
# =====================
DATA_ROOT = "/Users/admin/Desktop/AI/AH_01_playing/001/data/archive"
TEST_DIR = "Testing"
CKPT_PATH = "./runs_brain_mri/best_resnet18.pt"

IMG_SIZE = 224
BATCH_SIZE = 1      # Grad-CAM은 1장씩
NUM_SAMPLES = 6     # 시각화할 이미지 개수


# =====================
# Device
# =====================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =====================
# Grad-CAM class
# =====================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        score = logits[0, class_idx]
        score.backward()

        A = self.activations[0]   # [C,H,W]
        G = self.gradients[0]     # [C,H,W]

        weights = G.mean(dim=(1, 2))
        cam = torch.zeros(A.shape[1:], device=A.device)

        for c, w in enumerate(weights):
            cam += w * A[c]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), logits.detach()


# =====================
# Utils
# =====================
def denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def show_overlay(img, cam, title=""):
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = heatmap[..., ::-1] / 255.0

    overlay = 0.55 * img + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(5,5))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =====================
# Main
# =====================
def main():
    device = get_device()
    print("[Device]", device)

    # Transform
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Dataset / Loader
    test_ds = datasets.ImageFolder(
        os.path.join(DATA_ROOT, TEST_DIR),
        transform=tf
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}

    # Model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Grad-CAM (ResNet18 마지막 conv)
    cam_extractor = GradCAM(model, model.layer4)

    print(f"[Grad-CAM] visualize {NUM_SAMPLES} samples")

    for i, (x, y) in enumerate(test_loader):
        if i >= NUM_SAMPLES:
            break

        x = x.to(device)
        y_true = y.item()

        cam, logits = cam_extractor(x)
        pred = logits.argmax(1).item()

        img = denormalize(x[0])
        title = f"true={idx_to_class[y_true]} | pred={idx_to_class[pred]}"
        show_overlay(img, cam, title)

    cam_extractor.remove()


if __name__ == "__main__":
    main()
