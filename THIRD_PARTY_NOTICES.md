# 第三方来源

- 项目原有UFLD推理代码与模型结构来自[cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)。保留源码中的来源信息。
- `_vendor/CLRNet`来自[Turoad/CLRNet](https://github.com/Turoad/CLRNet)，保留其[Apache-2.0许可证](scripts/LanDetect/_vendor/CLRNet/LICENSE)。快照文件哈希在`docs/clrnet-snapshot.json`；不包含权重、编译产物或Git历史。
- `_vendor/shim`与`clrnet_runtime.py`是原项目保留的推理兼容层，覆盖有限的mmcv/addict/yapf/six接口及车道NMS，不等于安装了这些库的完整实现。后续替换时需复核模型权重键、预处理与NMS语义。
- FLUX、CLIP、T5、SegFormer等模型及KITTI/CODA数据遵守各自发布条款；仓库MIT许可不替代其许可。模型来源见`docs/models.md`。
- 本更新不含LoRA训练材料或模型二进制。
