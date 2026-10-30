# GigaPath factory/transform adapted from the official Apache-2.0 README example.
# Upstream review revision: 55431e04ff853a08f752a0ba42e6e9c48e60c776.
# Changes: registry/lambda assembly, img_size=224, shared normalization, eval().
# This public derivative omits the Virchow2 and TITAN/CONCH loader branches.
# See ../../docs/LICENSE_SCOPE.md and ../../licenses/Apache-2.0-GigaPath.txt.
# encoders.py
import timm
from . import config as C
from torchvision import transforms

IMAGENET_NORM = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

REGISTRY = {
  "gigapath": dict(  # E5 官方管线：256px@0.5mpp 提窗 → Resize(256,BICUBIC)+CenterCrop(224)，输出1536
    win_px=256, net_px=224, out_dim=1536,
    load=lambda: timm.create_model("hf_hub:prov-gigapath/prov-gigapath",
                                   pretrained=True, img_size=224).eval(),
    transform=transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(), IMAGENET_NORM]),
    postprocess=lambda out: out),
}

def read_px(win_px, mpp):
    """R-03 公式的唯一实现：READ=round(win_px×0.5/实际mpp)，禁止手填"""
    return round(win_px * C.TARGET_MPP / float(mpp))
