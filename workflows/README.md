# 工作流格式与依赖

根目录原有JSON及`legacy_xr`为历史ComfyUI界面工作流，需要相应节点、模型及输入。保留它们用于追溯，不将其旧参数套用于当前网页所有分支。

`api`由`python scripts/export_workflows.py`生成，使用`class_type`和`inputs`的ComfyUI API执行图格式；需放入JSON请求的`prompt`字段，不能将它当成界面编辑器的nodes/links文件。其输入前缀是`roadlab/example-类别`，提交前要准备匹配的图片、mask和blend。

网页运行直接调用Python图工厂，不从此目录读JSON。原生路线无需StarNodes及RoadBarrier LoRA，旧交通锥路线则需要二者。详见[部署](../docs/deployment.md)。
