# train_brain_mri_resnet18.py
# ------------------------------------------------------------
# Brain MRI 4-class classification (glioma/meningioma/notumor/pituitary)
# Fine-tuning (파인튜닝) with ResNet18
# - ImageFolder 로딩
# - Train/Val split (Training에서 분리) -> 데이터 누수 방지
# - tqdm 진행바
# - loss/accuracy 그래프 저장
# - best model 저장
# ------------------------------------------------------------

import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


@dataclass
class CFG:
    data_root: str = "/Users/admin/Desktop/AI/AH_01_playing/001/data/archive"
    train_dir: str = "Training"
    test_dir: str = "Testing"

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    epochs: int = 10
    lr: float = 0.0001
    weight_decay: float = 0.001

    val_ratio: float = 0.2
    seed: int = 42

    # fine-tuning strategy
    freeze_backbone: bool = True   # True면 backbone 얼리고 head만 학습
    unfreeze_epoch: int = 3        # 몇 epoch 이후에 backbone 일부(or 전체) 해제할지

    save_dir: str = "./runs_brain_mri"
    best_name: str = "best_resnet18.pt"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    # Mac: MPS, NVIDIA: CUDA, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(n * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return train_idx, val_idx

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            # fc(head)는 학습해야 하니까 제외
            if not name.startswith("fc."):
                p.requires_grad = False
    return model

def unfreeze_some(model: nn.Module):
    # 간단히: 전체 backbone 풀기 (원하면 layer4만 풀도록 바꿀 수 있음)
    for name, p in model.named_parameters():
        p.requires_grad = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter, acc_meter = 0.0, 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits, y)

        loss_meter += loss.item()
        acc_meter += acc

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    return loss_meter / len(loader), acc_meter / len(loader)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, desc="val"):
    model.eval()
    loss_meter, acc_meter = 0.0, 0.0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        acc = accuracy_from_logits(logits, y)

        loss_meter += loss.item()
        acc_meter += acc
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    return loss_meter / len(loader), acc_meter / len(loader)

def save_plot(history: Dict[str, List[float]], save_path: str):
    # loss plot
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_loss.png"))
    plt.show()
    plt.close()

    # acc plot
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_acc.png"))
    plt.show()
    plt.close()


@torch.no_grad()
def show_image_grid_from_loader(loader, idx_to_class, title="samples", n=8):
    """
    loader에서 배치 1개를 꺼내 n장만 그리드로 보여준다.
    return: None (화면 출력)
    """
    x, y = next(iter(loader))  # 첫 배치
    x = x[:n]
    y = y[:n]

    # Normalize 복원
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_vis = (x * std + mean).clamp(0, 1)

    grid = make_grid(x_vis, nrow=min(4, n))
    plt.figure(figsize=(10, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")

    labels = [idx_to_class[int(i)] for i in y]
    plt.title(f"{title} | labels: {labels}")
    plt.tight_layout()
    plt.show()


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = get_device()
    print(f"[Device] {device}")

    train_tf, eval_tf = make_transforms(cfg.img_size)

    train_path = os.path.join(cfg.data_root, cfg.train_dir)
    test_path = os.path.join(cfg.data_root, cfg.test_dir)

    # ✅ ImageFolder: 폴더명이 곧 클래스 라벨
    full_train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    # val은 transform이 달라야 하므로 "같은 파일 인덱스"로 eval transform dataset을 하나 더 만든다
    full_train_ds_eval = datasets.ImageFolder(train_path, transform=eval_tf)

    class_to_idx = full_train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    print("[Classes]", class_to_idx)

    train_idx, val_idx = split_indices(len(full_train_ds), cfg.val_ratio, cfg.seed)

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds_eval, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=False)

    # 최종 테스트용(Training에 섞지 않음 = 데이터 누수 방지)
    test_ds = datasets.ImageFolder(test_path, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=False)
    
    print("\n[Visual Check] show some samples (train/val/test)")
    show_image_grid_from_loader(train_loader, idx_to_class, title="TRAIN samples", n=8)
    show_image_grid_from_loader(val_loader, idx_to_class, title="VAL samples", n=8)
    show_image_grid_from_loader(test_loader, idx_to_class, title="TEST samples", n=8)


    model = build_model(num_classes=num_classes, freeze_backbone=cfg.freeze_backbone).to(device)

    # 학습 파라미터만 optimizer에 들어가게
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = os.path.join(cfg.save_dir, cfg.best_name)

    for epoch in range(1, cfg.epochs + 1):
        # freeze_backbone=True인 경우, 일정 epoch 이후 unfreeze해서 진짜 파인튜닝
        if cfg.freeze_backbone and epoch == cfg.unfreeze_epoch:
            print(f"[Unfreeze] epoch={epoch} -> unfreeze backbone")
            unfreeze_some(model)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay)  # 보통 더 작은 lr

        print(f"\nEpoch [{epoch}/{cfg.epochs}]")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device, desc="val")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"  train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"  val  : loss={va_loss:.4f}, acc={va_acc:.4f}")

        # best model 저장
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "cfg": cfg.__dict__,
                "best_val_acc": best_val_acc,
            }, best_path)
            print(f"  ✅ saved best -> {best_path} (val_acc={best_val_acc:.4f})")

    # 그래프 저장
    plot_path = os.path.join(cfg.save_dir, "history.png")
    save_plot(history, plot_path)
    print(f"[Saved plots] {plot_path.replace('.png','_loss.png')} , {plot_path.replace('.png','_acc.png')}")

    # best 로드 후 test 평가
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device, desc="test")
    print(f"\n[Test] loss={te_loss:.4f}, acc={te_acc:.4f}")

if __name__ == "__main__":
    main()


# ==========================
# Grad-CAM (어디를 보고 판단했는지)
# ==========================
class GradCAM:
    """
    ResNet18의 특정 레이어(feature map)에 대해 Grad-CAM을 계산한다.
    - return:
      - cam (H, W) numpy: 0~1로 정규화된 히트맵
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # forward hook: feature map 저장
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        # backward hook: gradient 저장
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # shape: [B, C, H, W]

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # shape: [B, C, H, W]

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """
        x: [1,3,H,W]
        class_idx: 타겟 클래스 인덱스(없으면 모델 예측 class 사용)
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # [1, num_classes]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # activations: [1, C, H, W], gradients: [1, C, H, W]
        A = self.activations[0]   # [C,H,W]
        G = self.gradients[0]     # [C,H,W]

        # channel-wise weights: GAP over gradients
        weights = G.mean(dim=(1, 2))  # [C]

        # cam = ReLU(sum_c w_c * A_c)
        cam = torch.zeros(A.shape[1:], device=A.device)  # [H,W]
        for c, w in enumerate(weights):
            cam += w * A[c]
        cam = torch.relu(cam)

        # normalize to 0~1
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), logits.detach()


def denormalize(img_tensor: torch.Tensor):
    """
    Normalize 된 텐서를 사람이 보는 이미지로 복원
    img_tensor: [3,H,W]
    return: [H,W,3] numpy in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def show_gradcam_overlay(img_rgb: np.ndarray, cam: np.ndarray, title: str = ""):
    """
    img_rgb: [H,W,3] in [0,1]
    cam: [H,W] in [0,1]
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.imshow(cam, alpha=0.45)  # alpha만으로도 충분히 "어디를 봤는지" 보임
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def pick_n_samples_from_loader(loader, n=6):
    """
    loader에서 n개 샘플을 뽑아 (x_batch, y_batch)로 반환
    """
    x, y = next(iter(loader))
    return x[:n], y[:n]


def run_gradcam_on_samples(model, loader, idx_to_class, device, n=6, target_layer=None):
    """
    - model이 어떤 영역을 보고 판단했는지 샘플 n개에 대해 Grad-CAM 시각화
    """
    model.eval()

    # ResNet18에서 보통 마지막 conv 블록(layer4)이 가장 보기 좋음
    if target_layer is None:
        target_layer = model.layer4  # ✅ 핵심

    cam_extractor = GradCAM(model, target_layer)

    x_batch, y_batch = pick_n_samples_from_loader(loader, n=n)

    for i in range(n):
        x = x_batch[i:i+1].to(device)  # [1,3,H,W]
        y_true = int(y_batch[i].item())

        cam, logits = cam_extractor(x, class_idx=None)  # None -> 예측 클래스 기준
        pred = int(torch.argmax(logits, dim=1).item())

        img_rgb = denormalize(x_batch[i])  # [H,W,3]
        title = f"true={idx_to_class[y_true]} | pred={idx_to_class[pred]}"
        show_gradcam_overlay(img_rgb, cam, title=title)

    cam_extractor.remove_hooks()