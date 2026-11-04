# 部署与配置

## Python依赖

Python 3.10或3.12用于CPU检查。先安装`requirements/core.txt`；自动车道路线增加`requirements/lane.txt`，语义保护增加`requirements/scene.txt`。PyTorch与torchvision需按目标平台安装匹配的CPU/CUDA构建；这里的依赖范围不是完整GPU环境锁文件。

## 文件位置

| 变量 | 默认值与作用 |
|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `6008`，远程部署按访问方式调整 |
| `COMFY_URL` | `http://127.0.0.1:8188` |
| `COMFY_ROOT` | 仓库下`ComfyUI`，仅用来推导输入/输出目录 |
| `COMFY_DATA` | 默认等于`COMFY_ROOT`，必须和ComfyUI实际输入/输出根目录一致 |
| `LANE_DIR` | 仓库下`scripts/LanDetect` |
| `ROADLAB_HOME` | 仓库下`.roadlab` |
| `ROADLAB_DATA` / `ROADLAB_CACHE` | 覆盖任务与缓存目录 |
| `ROADLAB_SCENE_MODEL` | 仓库下`models/segformer-b0-cityscapes` |
| `ROADLAB_PYTHON` | 仅供Linux启动脚本选择Python解释器 |

运行目录变化不会改变默认路径。Linux启动脚本读取仓库根`.env`，也可通过`ROADLAB_ENV_FILE`指定自己的配置。直接Python及Windows用户需在终端中设置环境变量。

## ComfyUI与模型

按[模型清单](models.md)准备权重，启动ComfyUI，并确认`/object_info`包含使用路线所需节点。网页与ComfyUI需共享同一份文件系统输入和输出目录，仅修改URL不足以连接一台没有共享目录的远程服务。

交通锥/垃圾桶路线依赖StarNodes `FluxFillSampler`；警示牌/行人依赖原生`CLIPTextEncodeFlux`、`ConditioningZeroOut`、`InpaintModelConditioning`、`KSampler`、`VAEDecode`和`ImageCompositeMasked`。

归档服务器ComfyUI提交记录为`f1b6bd43d7a6efdd9081cd191678ef35cc2052c4`。这是当时的环境记录，并不保证任意上游版本都兼容。任务取消使用`POST /api/jobs/{prompt_id}/cancel`按ID取消；部署的ComfyUI必须支持该接口，本项目不会退化成影响其他用户任务的全局中断。

## 启动与验证

```bash
python roadlab_web/app.py
# Linux使用.env时：bash roadlab_web/start.sh
curl 'http://127.0.0.1:6008/api/health?object_type=warning_sign&protect_original=0'
```

健康检查按类别返回缺失节点、权重或保护模型。模型未安装时页面可打开，但生成不会伪造替代结果。首次使用先以手动警示牌或行人验证，再验证车道与LoRA分支，以及按任务ID取消、历史恢复、保护开关和成员审核导出。

当前服务是研究用单用户/受控访问原型，未实现账号认证。需要远程使用时采用SSH转发或自行配置访问控制。前台启动适合调试，持续运行交由自己的服务管理器管理。
