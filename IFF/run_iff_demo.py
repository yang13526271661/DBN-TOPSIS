import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.dont_write_bytecode = True

from ds_iff_2026 import IFFConfig, LowAltitudeIFFRecognizer, RouteProfile
from iff_scene import generate_iff_time_series, state_to_observation


CSV_FIELDS = [
    "time",
    "target_id",
    "name",
    "truth",
    "label",
    "FR",
    "AC",
    "ST",
    "FO",
    "Theta",
    "Delta_H1",
    "Delta_V",
    "Delta_C",
    "Delta_H2",
    "window_size",
    "mean_conflict",
]


# 运行 IFF 场景并按目标维护滑动窗口识别结果。
def run_realtime_iff_assessment(num_steps=601, window_size=3):
    config = IFFConfig(
        route=RouteProfile(height_m=1000.0, speed_kmh=600.0, heading_deg=290.0),
        window_size=window_size,
    )
    recognizer = LowAltitudeIFFRecognizer(config)
    time_series, _friendly_series = generate_iff_time_series(num_steps=num_steps, config=config)

    windows = defaultdict(lambda: deque(maxlen=config.window_size))
    records = []

    for step in time_series:
        time_value = int(step[0]["Time"]) if step else 0
        target_results = []
        for state in step:
            obs = state_to_observation(state)
            windows[obs.target_id].append(obs)
            result = recognizer.identify(list(windows[obs.target_id]))
            target_results.append(_result_to_dict(result, obs.truth))
        records.append({"time": time_value, "targets": target_results})

    return records


# 将 IFF 实时识别结果写入 JSON 和 CSV 文件。
def write_results(records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "iff_realtime_results.json"
    csv_path = out_dir / "iff_realtime_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            for item in record["targets"]:
                writer.writerow(_csv_row(record, item))

    return json_path, csv_path


# 将指定时刻的 IFF 结果整理为终端摘要文本。
def summarize(records, final_time=None):
    record = records[-1] if final_time is None else next(r for r in records if r["time"] == final_time)
    lines = []
    for item in record["targets"]:
        mass = item["mass"]
        lines.append(
            f"T{item['target_id'] + 1:02d} {item['name']:<18} "
            f"truth={str(item.get('truth')):<4} label={item['label']:<2} "
            f"FR={mass['FR']:.3f} AC={mass['AC']:.3f} ST={mass['ST']:.3f} FO={mass['FO']:.3f} "
            f"Theta={mass['Theta']:.3f}"
        )
    return "\n".join(lines)


# 将 IFFResult 对象转换为可序列化字典。
def _result_to_dict(result, truth):
    return {
        "target_id": result.target_id,
        "name": result.name,
        "truth": truth,
        "label": result.label,
        "mass": result.mass,
        "deltas": result.deltas,
        "normalized_deltas": result.normalized_deltas,
        "window_size": result.window_size,
        "diagnostics": result.diagnostics,
    }


# 将单个目标结果转换为 CSV 行。
def _csv_row(record, item):
    mass = item["mass"]
    deltas = item["deltas"]
    return {
        "time": record["time"],
        "target_id": item["target_id"],
        "name": item["name"],
        "truth": item.get("truth"),
        "label": item["label"],
        "FR": mass["FR"],
        "AC": mass["AC"],
        "ST": mass["ST"],
        "FO": mass["FO"],
        "Theta": mass["Theta"],
        "Delta_H1": deltas.get("H1"),
        "Delta_V": deltas.get("V"),
        "Delta_C": deltas.get("C"),
        "Delta_H2": deltas.get("H2"),
        "window_size": item["window_size"],
        "mean_conflict": item["diagnostics"].get("mean_conflict", 0.0),
    }


# 解析 IFF demo 的命令行参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-style low-altitude IFF demo.")
    parser.add_argument("--steps", type=int, default=601, help="Number of scene time steps to simulate.")
    parser.add_argument("--window-size", type=int, default=3, help="Sliding evidence window size.")
    return parser.parse_args()


# 执行 demo 主流程并打印结果文件路径。
def main():
    args = parse_args()
    records = run_realtime_iff_assessment(num_steps=args.steps, window_size=args.window_size)
    out_dir = Path(__file__).resolve().parent / "results"
    json_path, csv_path = write_results(records, out_dir)
    print("IFF demo completed")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print("\nFinal-time summary:")
    print(summarize(records))


if __name__ == "__main__":
    main()
