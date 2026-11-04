# 批次统计导出

生成完成、达到尝试上限或收到正常停止请求时，自动在本批次的 `statistics/` 导出 12 张 CSV 和 `report.json`。
CSV 为 UTF-8 BOM，固定英文列名，中文类别名称。比例为 0–1，不是百分数。

## 手动重建

Blender「数据集」侧栏 → 设置“当前批次” → “导出统计表（CSV + JSON）”。不重新渲染。

也可在项目根运行：

```bash
python3 -I tools/generate/export_statistics.py data/batches/<批次目录>
```

正在写入的任务请先停止或完成后导出。手动重建只写 `statistics/`；面板另将导出结果追加到 `generation.log`。续跑后完整重算，绝不追加重复统计行。

## 文件

| 文件 | 口径 |
|---|---|
| summary.csv | 批次状态、目标/实际图片和板数、时长、配置、版本、源哈希、配额、数值最小/最大/均值/中位数 |
| splits.csv | train/val/test 的图片、板数和已记录场景组数 |
| classes.csv | 全部已解析标签的类别数量、占比、配额、差额（实际−配额） |
| colors.csv | 蓝/红/紫/无，紫色保留零行以便审计 |
| objects_per_image.csv | 每图有效解析标签行数的直方图，允许空标签背景为0 |
| distributions.csv | 颜色、尺度、姿态、遮挡、质量、位置、距离、高度、可见面积、增强事件等；unit区分object/image |
| cross_tables.csv | 每ID × 颜色/尺度/姿态/遮挡/质量/划分/区域/可见性/距离高度等桶；class_fraction以本ID总数为分母 |
| scene_groups.csv | 场景组图片清单、划分和跨集合泄漏标记 |
| coverage_gaps.csv | 缺桶、缺颜色素材、未验证质量条件 |
| checks.csv | 文件/标签/配额/组泄漏问题、各拒绝原因、独立渲染错误数和已记录程序错误 |
| images.csv | 每图一行，图片和标签相对路径、是否存在、尺寸、板数、增强、采样、耗时 |
| objects.csv | 每板一行，图片键(split+image_id)、plate_index、身份、几何参数、可见性、bbox、LT/LB/RB/RT坐标 |
| report.json | 同样的表、批次配置和数值摘要，程序读取版本schema_version=1 |

## 统计边界

- 总板数来自实际可解析的TXT行，不把图片数当目标数。`valid_label_objects`代表格式和有限数值可解析，不代表通过全部几何/视觉质检；必须同时看quality_check和checks.csv。
- `manifest_objects`是清单预期数量，`label_lines`是非空TXT行数；三者不一致会留在报告中。缺文件或非法行不伪造数据补足。
- 缺失历史字段以JSON null / CSV“未记录”表示；不拿主板的高度/距离冒充额外板的参数。
- 新批次逐板记录距离、实际底部离地高度、姿态、区域、可见面积；render_seconds为Cycles渲染耗时，export_seconds包含图像后处理与标签准备。active_seconds为本批次各次运行累计耗时，elapsed_seconds为首次启动到结束的墙钟时间。
- 老版error_count混合了拒绝与错误，保留为legacy_combined_error_count；没有独立render_error_count时不擅自解释为渲染失败数。
- 同一个被拒绝候选可能有多个原因，原因次数的合计可以大于被拒候选数。
- 图像质量clear/slight_blur/significant_blur按Cycles曝光端点位移分级；visible_moire仍未校准，不能由采样数推断。合成运动模糊尚未与实机曝光运动标定。图片表记录运动模糊启用、概率、曝光、速度和位移。
- 未完成批次complete=False；当前数量即便碰巧相等也不能冒充最终配额已完成。全批次平衡不等于train/val/test分别平衡。
- 缺少颜色素材的判断对新批次使用保存的asset_catalog_entries；历史未保存时参考当前catalog.json。
- 若质检发现错误，仍导出报告，quality_check=failed。报告生成失败仅记STATISTICS_ERROR，不删除图片、标签或场景。
- 停止按钮现在请求完成当前图后停止并导出统计；强杀/崩溃无法保证自动收尾，可手动对已有manifest补导出。

## 测试

```bash
python3 -I tools/generate/test_statistics.py
```

覆盖BOM/总数、只读重建、停止批次、非法与非有限标签、旧格式缺字段、空标签、空清单、组泄漏。实际停止/续跑测试脚本在 `tools/generate/runs/test_statistics_resume.py`，会另建一个小型测试批次。
