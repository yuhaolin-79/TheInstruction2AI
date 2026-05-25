"""
MNIST 手写数字识别 - 交互式 Web 演示
运行: python app.py
然后在浏览器打开 http://localhost:7860
"""

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import MNISTNet
from PIL import Image
import numpy as np
import os

# ===================== 加载模型 =====================
model = MNISTNet()
model_path = os.path.join(os.path.dirname(__file__), "mnist_model.pth")

if not os.path.exists(model_path):
    print("❌ 找不到模型文件！请先运行 python train.py")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()
print("✅ 模型加载成功")

# ===================== 预处理 =====================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def _extract_gray(image: np.ndarray) -> np.ndarray:
    """从 RGBA 画板数组提取白字黑底灰度图，兼容透明背景和白色背景两种情况"""
    alpha = image[:, :, 3]
    if alpha.min() < 128:
        # 透明背景：alpha 通道直接就是笔画 mask（笔画=255，背景=0）
        return alpha.astype(np.float32)
    else:
        # 不透明白色背景：用 RGB 灰度值反色得到白字黑底
        return 255.0 - np.mean(image[:, :, :3], axis=2)


def _center_digit(gray: np.ndarray) -> Image.Image | None:
    """裁剪笔画区域，缩放到 20x20，再用质心居中放入 28x28（与 MNIST 预处理一致）"""
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    coords = np.argwhere(gray > 30)
    if len(coords) == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = gray[y0:y1 + 1, x0:x1 + 1].astype(np.float32)

    # 缩放到 20x20 以内（保持宽高比），与 MNIST 原始预处理一致
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    digit = np.array(
        Image.fromarray(cropped.astype(np.uint8), mode="L").resize(
            (new_w, new_h), Image.LANCZOS
        ),
        dtype=np.float32,
    )

    # 计算质心，将质心对齐到 28x28 画布中心（14, 14）
    total = digit.sum()
    if total == 0:
        return None
    cy = float((np.arange(new_h) @ digit.sum(axis=1)) / total)
    cx = float((np.arange(new_w) @ digit.sum(axis=0)) / total)

    canvas = np.zeros((28, 28), dtype=np.float32)
    oy = int(round(14 - cy))
    ox = int(round(14 - cx))
    sy0, sy1 = max(0, -oy), min(new_h, 28 - oy)
    sx0, sx1 = max(0, -ox), min(new_w, 28 - ox)
    dy0, dy1 = sy0 + oy, sy1 + oy
    dx0, dx1 = sx0 + ox, sx1 + ox
    if sy1 > sy0 and sx1 > sx0:
        canvas[dy0:dy1, dx0:dx1] = digit[sy0:sy1, sx0:sx1]

    return Image.fromarray(canvas.astype(np.uint8), mode="L")


def predict(image):
    empty = {str(i): 0.0 for i in range(10)}
    if image is None:
        return empty, None

    if isinstance(image, dict):
        image = image.get("composite", image)
    if not isinstance(image, np.ndarray):
        return empty, None

    # 打印诊断信息，帮助排查通道问题
    alpha = image[:, :, 3]
    print(f"[debug] alpha min={alpha.min()} max={alpha.max()} | "
          f"RGB mean={np.mean(image[:,:,:3]):.1f}")

    gray = _extract_gray(image)
    img_28 = _center_digit(gray)
    if img_28 is None:
        return empty, None

    # 缩放到 28x28 用于预览和推理
    preview = img_28.resize((140, 140), Image.NEAREST)

    tensor = transform(img_28).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)[0]

    return {str(i): float(probabilities[i]) for i in range(10)}, np.array(preview)


# ===================== Gradio 界面 =====================
with gr.Blocks(title="MNIST 手写数字识别") as demo:
    gr.Markdown("# ✍️ MNIST 手写数字识别", elem_classes="main-title")
    gr.Markdown(
        "在左侧画板上写一个数字 (0-9)，模型会实时给出预测概率",
        elem_classes="sub-title",
    )

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="在这里写数字",
                type="numpy",
                image_mode="RGBA",
                canvas_size=(280, 280),
                brush=gr.Brush(default_size=24, colors=["#000000"]),
                elem_id="sketchpad-wrap",
            )
            _clear_count = gr.State(value=0)
            clear_btn = gr.Button("🗑️ 清除画板")
            clear_btn.click(
                fn=lambda n: (None, n + 1),
                inputs=_clear_count,
                outputs=[canvas, _clear_count],
            ).then(
                fn=None,
                js="""() => {
                    setTimeout(() => {
                        for (const label of ['Draw','draw','Pencil','pencil','Brush','brush']) {
                            const btn = document.querySelector('[aria-label="' + label + '"]');
                            if (btn) { btn.click(); return; }
                        }
                        const wrap = document.getElementById('sketchpad-wrap');
                        if (wrap) {
                            const btns = [...wrap.querySelectorAll('button')]
                                .filter(b => b.querySelector('svg'));
                            if (btns[0]) btns[0].click();
                        }
                    }, 250);
                }"""
            )

        with gr.Column(scale=1):
            label_output = gr.Label(label="预测结果", num_top_classes=5)
            model_view = gr.Image(label="模型实际看到的图像（28×28）",
                                  image_mode="L", width=140, height=140)

    # 实时预测
    canvas.change(fn=predict, inputs=canvas, outputs=[label_output, model_view])

    gr.Markdown("---")
    gr.Markdown(
        "**技术栈**: PyTorch CNN · Gradio · MNIST 数据集 (60k训练 / 10k测试)  \n"
        "**模型**: 2层卷积 + 2层全连接, ~63k 参数, 测试准确率 ~99%"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=".main-title{text-align:center;margin-bottom:.5em}.sub-title{text-align:center;color:#666;margin-bottom:1.5em}",
    )
