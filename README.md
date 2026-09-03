# PhotoCut

PhotoCut 用来批量裁剪扫描照片：自动检测照片四角，在本机浏览器中确认或调整，然后完成透视校正和裁剪。所有照片都只在本机处理。

## 安装

需要 Python 3.10 或更高版本，支持 macOS、Windows 和 Linux。

```bash
git clone https://github.com/LittleSixNine/photocut.git
cd photocut
python3 -m venv .venv
source .venv/bin/activate        # Windows：.venv\Scripts\activate
python3 -m pip install -e .
```

仓库已经包含可以再分发的 V8.2 ONNX 模型，不需要另外下载模型，默认的 PhotoCut Selector v4 可以直接运行。

## 使用

将下面的 `input` 换成你的照片文件夹，依次执行三个步骤：

```bash
# 1. 检测四角
photocut input --detect

# 2. 在本机浏览器中确认或调整
photocut input --confirm

# 3. 透视校正并裁剪
photocut input --crop
```

默认输出到输入文件夹旁边的 `<文件夹名>裁剪/`。裁剪时会向内缩进 50 个原图像素以去除扫描白边，源照片不会被修改。

默认场景是一张放在白色扫描仪底面上的实体照片。其他单照片背景可使用：

```bash
photocut input --scene-profile generic_single --detect
```

也可以继续使用兼容入口 `python3 photocut_cli.py ...`。

## 版本

- PhotoCut Selector v4：默认选择和编排逻辑；历史记录中的机器标识仍为 `auto-v4`
- V8.2：默认白色扫描底检测器
- V7.1：候选生成和安全回退
- v5.2：兼容检测器
- GUI 2.0：默认的本机浏览器确认界面

## 测试

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 scripts/check_public_tree.py
```

## 许可

PhotoCut 源码和随项目提供的 V8.2 模型使用 MIT License。模型使用了由 TorchVision 权重初始化的 MobileNetV3 backbone，随模型保留了 BSD-3-Clause 上游声明。
