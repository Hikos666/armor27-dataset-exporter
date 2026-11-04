"""Batch-wide object quotas. Counts refer to every plate, not every image."""

import math
import random
from collections import Counter


def make_plan(count, minimum, maximum, distribution, class_ids, lamda, seed, group_size=1):
    classes = sorted(set(class_ids))
    if not classes:
        raise ValueError("没有可生成的类别")
    if not math.isfinite(lamda) or not 0 <= lamda <= 1:
        raise ValueError("λ 必须在 0 到 1 之间")
    if count < 1 or minimum < 1 or maximum < minimum:
        raise ValueError("图片数量必须大于零，且 1 ≤ 最少板数 ≤ 最多板数")
    if distribution not in ("uniform", "fewer", "more"):
        raise ValueError("未知的同框数量分布")
    k = len(classes)
    if lamda == 0 and math.ceil(count * minimum / k) * k > count * maximum:
        raise ValueError(
            f"λ=0 无法严格平衡：{count} 张图、每图 {minimum}–{maximum} 块，"
            f"总板数范围 {count * minimum}–{count * maximum} 内没有 {k} 的倍数。"
            "请修改图片数量或每图板数范围。")
    # Separate from pose RNG so rejections cannot change the quotas or image counts.
    rng = random.Random(seed)
    weights = [1 if distribution == "uniform" else
               (v - minimum + 1 if distribution == "more" else maximum - v + 1)
               for v in range(minimum, maximum + 1)]
    if group_size < 1:
        raise ValueError("场景组大小必须为正数")
    sizes = [min(group_size, count - start) for start in range(0, count, group_size)]
    group_counts = rng.choices(range(minimum, maximum + 1), weights=weights, k=len(sizes))
    original_total = sum(size * n for size, n in zip(sizes, group_counts))
    if lamda == 0:
        # Residue DP: each scene group has identical object count across its ID swaps.
        # Minimize adjustments while making the final total divisible by class count.
        history, costs = [], {0: 0}
        for size, original in zip(sizes, group_counts):
            next_costs, choices = {}, {}
            for residue, cost in costs.items():
                for n in range(minimum, maximum + 1):
                    after = (residue + size * n) % k
                    candidate = cost + size * abs(n - original)
                    if after not in next_costs or candidate < next_costs[after]:
                        next_costs[after] = candidate
                        choices[after] = (residue, n)
            history.append(choices)
            costs = next_costs
        if 0 not in costs:
            raise ValueError("同场景组大小与板数范围无法满足严格平衡，请调整图片数量或板数范围")
        residue = 0
        for index in reversed(range(len(sizes))):
            residue, group_counts[index] = history[index][residue]
    counts, scene_groups = [], []
    for group_id, (size, n) in enumerate(zip(sizes, group_counts)):
        counts.extend([n] * size)
        scene_groups.extend([group_id] * size)
    total = sum(counts)
    balanced_each = total // k if lamda == 0 else math.floor((1 - lamda) * total / k)
    schedule = [class_id for class_id in classes for _ in range(balanced_each)]
    schedule.extend(rng.choice(classes) for _ in range(total - len(schedule)))
    rng.shuffle(schedule)
    quotas = Counter(schedule)
    if lamda == 0 and group_size > 1:
        # Every complete K-image scene group sees all IDs in every object slot.
        remaining = quotas.copy()
        schedule = []
        for size, n in zip(sizes, group_counts):
            if size == k:
                columns = []
                for _ in range(n):
                    column = list(classes)
                    rng.shuffle(column)
                    columns.append(column)
                for row in range(size):
                    for column in columns:
                        schedule.append(column[row])
                        remaining[column[row]] -= 1
            else:
                for _ in range(size * n):
                    pool = [c for c in classes if remaining[c] > 0]
                    chosen = rng.choice(pool)
                    schedule.append(chosen)
                    remaining[chosen] -= 1
    return {
        "version": 1, "lamda": lamda, "seed": seed, "classes": classes,
        "image_counts": counts, "schedule": schedule,
        "scene_groups": scene_groups, "group_size": group_size,
        "quotas": {str(c): quotas[c] for c in classes},
        "total_objects": total, "unadjusted_total": original_total,
        "balanced_each": balanced_each,
        "count_distribution": distribution, "minimum": minimum, "maximum": maximum,
    }


def sample_objects(sample):
    return sample.get("plates") or [sample]


def actual_counts(samples):
    return Counter(plate["class_id"] for sample in samples for plate in sample_objects(sample))


def verify_prefix(plan, samples):
    if len(samples) > len(plan["image_counts"]):
        raise ValueError("已有图片数量超过配额计划")
    actual = []
    for index, sample in enumerate(samples):
        plates = sample_objects(sample)
        if len(plates) != plan["image_counts"][index]:
            raise ValueError(f"第 {index + 1} 张图片的板数与配额计划不符")
        actual.extend(plate["class_id"] for plate in plates)
    if actual != plan["schedule"][:len(actual)]:
        raise ValueError("已完成标签与类别配额检查点不一致")
    return len(actual)


def entries_for_image(plan, samples, entries, rng):
    offset = verify_prefix(plan, samples)
    number = len(samples)
    wanted = plan["image_counts"][number]
    by_class = {}
    for entry in entries:
        by_class.setdefault(entry["class_id"], []).append(entry)
    # First select the class from the schedule, then choose a color/material within it.
    # A class with three materials does not get three times the probability.
    return [rng.choice(by_class[c]) for c in plan["schedule"][offset:offset + wanted]]
