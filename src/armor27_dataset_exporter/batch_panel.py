"""Blender sidebar for the armor27 batch generator.

View3D > Sidebar > 数据集. Start launches a separate background Blender so the
open file is not saved or replaced. Stop writes a STOP file and terminates the
process. Resume continues a batch that has rng_state.pkl.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent
STATE_ROOT = Path(os.environ.get(
    "ARMOR27_STATE_ROOT", Path.home() / "New_space" / "armor27_dataset" / "state"
)).expanduser().resolve()
JOB = STATE_ROOT / "current_job.json"
ASSET_ROOT_DEFAULT = ROOT
OUTPUT_ROOT_DEFAULT = Path.home() / "New_space" / "armor27_dataset" / "batches"
CLASS_NAMES = ("哨兵", "1", "2", "3", "4", "5", "前哨", "基地")
COLOR_FIELDS = ((0, "蓝"), (1, "红"), (3, "无"))

bl_info = {
    "name": "Armor27 Batch",
    "author": "armor27",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > 数据集",
    "description": "Generate, stop, resume, and check armor plate datasets",
    "category": "Render",
}


def _python():
    """Quality check and statistics use only the standard library, but they read
    sys.argv directly, so they must run under a real Python, not the Blender binary
    (which is what sys.executable points to inside an add-on)."""
    python = shutil.which("python3")
    if python is None:
        raise RuntimeError("质量检查和统计导出需要系统的 python3，但没有找到")
    return python


def _job():
    if not JOB.is_file():
        return None
    try:
        return json.loads(JOB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _running(job):
    if not job or not job.get("pid"):
        return False
    try:
        os.kill(job["pid"], 0)
        stat = Path(f"/proc/{job['pid']}/stat")
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
    except OSError:
        return False
    return True


def _write_job(payload):
    JOB.parent.mkdir(parents=True, exist_ok=True)
    JOB.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _launch(batch, resume):
    settings = bpy.context.window_manager.armor27_batch
    if _running(_job()):
        raise RuntimeError("已有生成任务在运行")
    classes = [i for i in range(8) if getattr(settings, f"class_{i}")]
    colors = [index for index, _name in COLOR_FIELDS if getattr(settings, f"color_{index}")]
    if not classes or not colors:
        raise RuntimeError("至少选择一个编号和一个颜色")
    asset_root = Path(bpy.path.abspath(settings.asset_root)).expanduser().resolve()
    bindings = json.loads(
        (ROOT / "tools" / "asset_validation" / "asset_bindings.json").read_text(encoding="utf-8"))
    missing = [name for name in (bindings["source_blend"], "IDs") if not (asset_root / name).exists()]
    if missing:
        raise RuntimeError(f"资源目录缺少 {', '.join(missing)}：{asset_root}")
    if settings.min_plates > settings.max_plates:
        raise RuntimeError("最少板数不能大于最多板数")
    if not resume:
        catalog = json.loads((ROOT / "tools" / "generate" / "catalog.json").read_text(encoding="utf-8"))
        active = {e["class_id"] for e in catalog["entries"]
                  if e["class_id"] in classes and e["color_id"] in colors and e["color_id"] != 2}
        if not active:
            raise RuntimeError("没有符合所选编号和颜色的素材")
        lower, upper, k = settings.count * settings.min_plates, settings.count * settings.max_plates, len(active)
        if settings.lamda == 0 and ((lower + k - 1) // k) * k > upper:
            raise RuntimeError(f"λ=0 要求总板数是 {k} 的倍数；当前范围 {lower}–{upper} 无法满足，请调整数量")
    batch.mkdir(parents=True, exist_ok=True)
    command = [
        bpy.app.binary_path, "--background", "--python", str(ROOT / "tools" / "generate" / "generate_batch.py"),
        "--", "--batch", str(batch), "--count", str(settings.count), "--seed", str(settings.seed),
        "--train", str(settings.train_ratio), "--val", str(settings.val_ratio),
        "--test", str(settings.test_ratio), "--debug", "1" if settings.debug_images else "0",
        "--samples", str(settings.render_samples),
        "--motion-blur-probability", str(settings.motion_blur_probability),
        "--classes", ",".join(str(i) for i in range(8) if getattr(settings, f"class_{i}")),
        "--colors", ",".join(str(index) for index, _name in COLOR_FIELDS if getattr(settings, f"color_{index}")),
        "--min-plates", str(settings.min_plates),
        "--max-plates", str(settings.max_plates),
        "--plate-distribution", settings.plate_distribution,
        "--lamda", str(settings.lamda),
    ]
    if resume:
        command.append("--resume")
    log_handle = (batch / "process.log").open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["ARMOR27_ASSET_ROOT"] = str(Path(bpy.path.abspath(settings.asset_root)).expanduser().resolve())
    environment["ARMOR27_OUTPUT_ROOT"] = str(Path(bpy.path.abspath(settings.output_dir)).expanduser().resolve())
    process = subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True, env=environment)
    settings.batch_dir = str(batch)
    _write_job({"pid": process.pid, "batch": str(batch), "started": datetime.now().isoformat(timespec="seconds")})
    return process.pid


_PHASE = {
    "loading": "正在读取场景",
    "running": "正在导出",
    "passed": "已完成",
    "stopped": "已停止",
    "needs_review": "未达到数量",
}
_SPLIT = {"train": "训练", "val": "验证", "test": "测试"}


def _export_progress(batch_dir, fallback_count):
    """Read export progress from the batch the generator is already writing."""
    if not batch_dir:
        return None
    root = Path(batch_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if (root / "generation.log").is_file():
            return {"done": 0, "target": fallback_count, "phase": "loading",
                    "attempts": 0, "rejected": 0, "splits": {}}
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    samples = manifest.get("samples") or []
    plan = manifest.get("split_plan") or []
    splits = {}
    for sample in samples:
        name = sample.get("split") or "?"
        splits[name] = splits.get(name, 0) + 1
    return {
        "done": manifest.get("sample_count", len(samples)),
        "target": len(plan) or fallback_count,
        "phase": manifest.get("status") or "running",
        "attempts": manifest.get("attempts", 0),
        "rejected": len(manifest.get("rejected") or []),
        "splits": splits,
        "object_class_counts": manifest.get("object_class_counts", {}),
        "class_quotas": (manifest.get("balance_plan") or {}).get("quotas", {}),
    }


def _tick():
    try:
        manager = bpy.context.window_manager
        if manager is None:
            return 1.0
        for window in manager.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception:
        return 2.0
    return 1.0


class Armor27Settings(bpy.types.PropertyGroup):
    count: bpy.props.IntProperty(name="生成数量", default=8, min=1, max=100000)
    seed: bpy.props.IntProperty(name="随机种子", default=27, min=0)
    train_ratio: bpy.props.FloatProperty(name="训练", default=0.75, min=0.0, max=1.0, precision=2)
    val_ratio: bpy.props.FloatProperty(name="验证", default=0.25, min=0.0, max=1.0, precision=2)
    test_ratio: bpy.props.FloatProperty(name="测试", default=0.0, min=0.0, max=1.0, precision=2)
    asset_root: bpy.props.StringProperty(
        name="资源目录", subtype="DIR_PATH", default=str(ASSET_ROOT_DEFAULT),
        description="包含源 blend 文件和 IDs 目录的资源目录",
    )
    output_dir: bpy.props.StringProperty(
        name="输出目录", subtype="DIR_PATH", default=str(OUTPUT_ROOT_DEFAULT))
    debug_images: bpy.props.BoolProperty(name="生成调试图", default=True)
    render_samples: bpy.props.IntProperty(name="渲染采样", default=32, min=1, max=4096)
    motion_blur_probability: bpy.props.FloatProperty(
        name="运动模糊比例", default=0.25, min=0.0, max=1.0, precision=2,
        description="每张图/同场景组启用真实Cycles曝光运动模糊的概率。0关闭，1全部启用",
    )
    lamda: bpy.props.FloatProperty(
        name="类别随机系数 λ", default=0.0, min=0.0, max=1.0, precision=3,
        description="0：每类总板数严格相等；1：每块板等概率随机选类别。按所有板计数，不按图片计数",
    )
    class_0: bpy.props.BoolProperty(name="哨兵", default=True)
    class_1: bpy.props.BoolProperty(name="1", default=True)
    class_2: bpy.props.BoolProperty(name="2", default=True)
    class_3: bpy.props.BoolProperty(name="3", default=True)
    class_4: bpy.props.BoolProperty(name="4", default=True)
    class_5: bpy.props.BoolProperty(name="5", default=True)
    class_6: bpy.props.BoolProperty(name="前哨", default=True)
    class_7: bpy.props.BoolProperty(name="基地", default=True)
    color_0: bpy.props.BoolProperty(name="蓝", default=True)
    color_1: bpy.props.BoolProperty(name="红", default=True)
    color_3: bpy.props.BoolProperty(name="无", default=True)
    min_plates: bpy.props.IntProperty(name="最少", default=1, min=1, max=6)
    max_plates: bpy.props.IntProperty(name="最多", default=1, min=1, max=6)
    plate_distribution: bpy.props.EnumProperty(
        name="分布",
        items=(
            ("uniform", "均匀", "最少到最多之间等概率"),
            ("fewer", "偏少", "更常出现较少的装甲板"),
            ("more", "偏多", "更常出现较多的装甲板"),
        ),
        default="uniform",
    )
    batch_dir: bpy.props.StringProperty(name="当前批次", default="")


class BATCH_OT_start(bpy.types.Operator):
    bl_idname = "armor27.start_batch"
    bl_label = "开始生成"
    bl_description = "在后台读取已保存的 blend 开始新批次，不写回当前文件"

    def execute(self, context):
        settings = context.window_manager.armor27_batch
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        batch = Path(bpy.path.abspath(settings.output_dir)) / stamp
        try:
            pid = _launch(batch, resume=False)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"已开始 {batch.name} pid={pid}")
        return {"FINISHED"}


class BATCH_OT_stop(bpy.types.Operator):
    bl_idname = "armor27.stop_batch"
    bl_label = "停止生成"
    bl_description = "完成当前图后停止，保留续跑检查点并导出统计"

    def execute(self, context):
        job = _job()
        if not job:
            self.report({"WARNING"}, "没有正在运行的任务")
            return {"CANCELLED"}
        batch = Path(job["batch"])
        (batch / "STOP").write_text("stop\n", encoding="utf-8")
        self.report({"INFO"}, "已请求停止：当前图片完成后导出统计")
        return {"FINISHED"}


class BATCH_OT_resume(bpy.types.Operator):
    bl_idname = "armor27.resume_batch"
    bl_label = "继续生成"
    bl_description = "从当前批次已完成的样本和随机数检查点继续"

    def execute(self, context):
        settings = context.window_manager.armor27_batch
        batch = Path(bpy.path.abspath(settings.batch_dir)) if settings.batch_dir else None
        if batch is None or not (batch / "manifest.json").is_file():
            root = Path(bpy.path.abspath(settings.output_dir))
            candidates = sorted(root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime)
            if not candidates:
                self.report({"ERROR"}, "没有可继续的批次")
                return {"CANCELLED"}
            batch = candidates[-1].parent
        try:
            pid = _launch(batch, resume=True)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"继续 {batch.name} pid={pid}")
        return {"FINISHED"}


class BATCH_OT_check(bpy.types.Operator):
    bl_idname = "armor27.check_batch"
    bl_label = "运行质量检查"
    bl_description = "检查当前批次的图片、17 字段标签和角点几何"

    def execute(self, context):
        settings = context.window_manager.armor27_batch
        batch = settings.batch_dir
        if not batch:
            self.report({"ERROR"}, "还没有批次")
            return {"CANCELLED"}
        try:
            command = [_python(), "-I", str(ROOT / "tools" / "generate" / "check_dataset.py"), batch]
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        result = subprocess.run(command, capture_output=True, text=True)
        log = Path(batch) / "generation.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(result.stdout or "")
            if result.stderr:
                handle.write(result.stderr)
        if result.returncode != 0:
            self.report({"ERROR"}, "质量检查未通过，详见 generation.log")
            return {"CANCELLED"}
        self.report({"INFO"}, "质量检查通过")
        return {"FINISHED"}


class BATCH_OT_statistics(bpy.types.Operator):
    bl_idname = "armor27.export_statistics"
    bl_label = "导出统计表"
    bl_description = "为当前已完成或停止的批次导出CSV和JSON，不重新渲染"

    def execute(self, context):
        if _running(_job()):
            self.report({"WARNING"}, "请等待本批次完成或停止后导出统计")
            return {"CANCELLED"}
        batch = context.window_manager.armor27_batch.batch_dir
        if not batch or not (Path(batch) / "manifest.json").is_file():
            self.report({"ERROR"}, "请指定包含manifest.json的批次目录")
            return {"CANCELLED"}
        try:
            command = [_python(), "-I", str(ROOT / "tools" / "generate" / "export_statistics.py"), batch]
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        result = subprocess.run(command, capture_output=True, text=True)
        with (Path(batch) / "generation.log").open("a", encoding="utf-8") as handle:
            handle.write(result.stdout or "")
            handle.write(result.stderr or "")
        if result.returncode:
            self.report({"ERROR"}, "统计导出失败，详见generation.log；原数据未改动")
            return {"CANCELLED"}
        self.report({"INFO"}, "已导出到当前批次的statistics目录")
        return {"FINISHED"}


class BATCH_PT_panel(bpy.types.Panel):
    bl_label = "批量生成"
    bl_idname = "BATCH_PT_armor27"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "数据集"

    def draw(self, context):
        layout = self.layout
        settings = getattr(context.window_manager, "armor27_batch", None)
        if settings is None:
            layout.label(text="插件设置尚未注册，请重新启用插件")
            return
        running = _running(_job())
        layout.label(text="生成中" if running else "空闲")
        layout.prop(settings, "count")
        layout.prop(settings, "seed")
        layout.prop(settings, "render_samples")
        layout.label(text="编号")
        row = layout.row(align=True)
        for index, name in enumerate(CLASS_NAMES):
            row.prop(settings, f"class_{index}", toggle=True, text=name)
        layout.label(text="颜色")
        row = layout.row(align=True)
        for index, name in COLOR_FIELDS:
            row.prop(settings, f"color_{index}", toggle=True, text=name)
        layout.label(text="同框装甲板")
        row = layout.row(align=True)
        row.prop(settings, "min_plates")
        row.prop(settings, "max_plates")
        layout.prop(settings, "plate_distribution")
        layout.prop(settings, "lamda", slider=True)
        layout.prop(settings, "motion_blur_probability")
        layout.label(text="λ=0 严格平衡，λ=1 随机类别")
        row = layout.row(align=True)
        row.prop(settings, "train_ratio")
        row.prop(settings, "val_ratio")
        row.prop(settings, "test_ratio")
        layout.prop(settings, "asset_root")
        layout.prop(settings, "output_dir")
        layout.prop(settings, "debug_images")
        layout.prop(settings, "batch_dir")
        row = layout.row(align=True)
        row.enabled = not running
        row.operator("armor27.start_batch", text="开始生成")
        row.operator("armor27.resume_batch", text="继续生成")
        layout.operator("armor27.stop_batch", text="停止生成")
        layout.operator("armor27.check_batch", text="运行质量检查")
        layout.operator("armor27.export_statistics", text="导出统计表（CSV + JSON）")
        layout.label(text="读取已保存的 blend，不写回当前文件。紫色不生成。")
        batch_dir = settings.batch_dir or ((_job() or {}).get("batch") or "")
        progress = _export_progress(batch_dir, settings.count)
        if batch_dir and not running:
            process_log = Path(batch_dir) / "process.log"
            if process_log.is_file():
                with process_log.open("rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 8192))
                    tail = handle.read().decode("utf-8", errors="replace")
                failures = [line for line in tail.splitlines()
                            if line.startswith(("RuntimeError:", "ValueError:", "Traceback"))]
                if failures:
                    box = layout.box()
                    box.alert = True
                    box.label(text="后台任务异常退出，请查看 process.log")
                    box.label(text=failures[-1][-100:])
        if progress and progress["target"]:
            done = progress["done"]
            target = progress["target"]
            factor = min(1.0, done / target)
            box = layout.box()
            box.label(text="导出进度")
            try:
                box.progress(factor=factor, type="BAR", text=f"{done} / {target}")
            except Exception:
                box.label(text=f"{done} / {target}")
            box.label(text=_PHASE.get(progress["phase"], progress["phase"]))
            split_text = "  ".join(f"{_SPLIT.get(name, name)} {count}" for name, count in progress["splits"].items())
            if split_text:
                box.label(text=split_text)
            box.label(text=f"尝试 {progress['attempts']}，拒绝 {progress['rejected']}")
            counts = progress.get("object_class_counts", {})
            for class_id, quota in progress.get("class_quotas", {}).items():
                name = CLASS_NAMES[int(class_id)]
                box.label(text=f"{name}：{counts.get(class_id, 0)} / {quota} 块")
        if settings.batch_dir:
            log_path = Path(settings.batch_dir) / "generation.log"
            if log_path.is_file():
                box = layout.box()
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
                    box.label(text=line[-80:])


CLASSES = (
    Armor27Settings, BATCH_OT_start, BATCH_OT_stop, BATCH_OT_resume, BATCH_OT_check, BATCH_OT_statistics, BATCH_PT_panel,
)


def register():
    unregister()
    # Assign the settings property before the panel. Blender redraws the sidebar
    # as soon as the panel class is registered, and draw() runs before register() returns.
    bpy.utils.register_class(Armor27Settings)
    bpy.types.WindowManager.armor27_batch = bpy.props.PointerProperty(type=Armor27Settings)
    for cls in CLASSES:
        if cls is not Armor27Settings:
            bpy.utils.register_class(cls)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if hasattr(bpy.types.WindowManager, "armor27_batch"):
        del bpy.types.WindowManager.armor27_batch
    for cls in reversed(CLASSES):
        if hasattr(bpy.types, cls.__name__):
            bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
