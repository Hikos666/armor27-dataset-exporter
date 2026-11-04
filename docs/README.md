# Armor27 数据集导出工具

这是一个 Blender 插件，用于生成装甲板合成数据集、停止/恢复批次、执行质量检查并导出统计表。

## 系统要求

- Blender 5.2.0 或更高版本
- Linux 首发验证环境
- Full 包约 104 MB，包含源 `.blend` 场景和 `IDs/` 图片

## 包类型

- `*-addon.zip`：插件源码和配置。安装后需要在面板中选择外部资源目录。
- `*-full.zip`：插件源码、配置、场景文件和 ID 图片，可直接使用。

## 安装

1. 打开 Blender。
2. 进入“编辑 → 偏好设置 → 插件”。
3. 选择“从磁盘安装”。
4. 选择发布包中的 `*-addon.zip` 或 `*-full.zip`。
5. 启用 `Armor27 Dataset Exporter`。
6. 打开 3D 视图侧栏的“数据集”标签。

Full 包默认把资源目录设置为插件目录。Addon 包需要把“资源目录”设置为包含以下内容的目录：

```text
<资源目录>/armor_newnewnew.blend
<资源目录>/IDs/*.jpg
```

## 使用

1. 选择资源目录和输出目录。
2. 选择生成数量、类别、颜色和数据集比例。
3. 点击“开始生成”。
4. 可用“停止生成”创建可恢复检查点。
5. 使用“继续生成”恢复包含 `manifest.json` 的批次。
6. 完成后运行质量检查或导出统计表。

后台任务使用当前 Blender 的 Python/二进制路径，不依赖发布机器上的固定路径。生成器只读源 `.blend`，不会保存覆盖它。

## 输出

每个批次目录包含：

- `images/{train,val,test}/`
- `labels/{train,val,test}/`
- `manifest.json`
- `generation.log`
- `statistics/report.json` 及 CSV 表
- 可选的 `preview/`

## 命令行环境变量

- `ARMOR27_ASSET_ROOT`：资源目录
- `ARMOR27_OUTPUT_ROOT`：未从命令行指定批次时的默认输出根目录
- `ARMOR27_STATE_ROOT`：插件任务状态文件目录

## 构建

在仓库根目录运行：

```bash
python3 tools/dataset_exporter_release/build_release.py
```

构建结果位于 `tools/dataset_exporter_release/build/`，同时生成 SHA-256 校验文件。
