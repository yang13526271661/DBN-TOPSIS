import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA = Path("results_fig/scene_B_deception/visual_data.json")
DEFAULT_OUTPUT = Path("results_fig/scene_B_deception/T2_intent_posterior.png")
DEFAULT_TABLE_TIMES = (120, 170, 240, 270)


def find_target(record, label):
    return next(
        (enemy for enemy in record.get("enemies", []) if enemy.get("label") == label),
        None,
    )


def load_records(data_path):
    with data_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No visualization records found in {data_path}")
    return records


def collect_posterior_series(
    records,
    target_label,
    source_intent,
    target_intent,
    start_time,
    end_time,
):
    times = []
    source_probabilities = []
    target_probabilities = []

    for record in records:
        time_value = int(record.get("time", 0))
        if time_value < start_time or time_value > end_time:
            continue

        target = find_target(record, target_label)
        if target is None:
            continue
        probabilities = target.get("intent_probabilities")
        if not isinstance(probabilities, dict):
            continue

        source_probability = probabilities.get(source_intent)
        target_probability = probabilities.get(target_intent)
        if source_probability is None or target_probability is None:
            continue

        times.append(time_value)
        source_probabilities.append(float(source_probability))
        target_probabilities.append(float(target_probability))

    if not times:
        raise ValueError(
            f"No posterior data found for {target_label} between "
            f"{start_time}s and {end_time}s"
        )

    return (
        np.asarray(times, dtype=float),
        np.asarray(source_probabilities, dtype=float),
        np.asarray(target_probabilities, dtype=float),
    )


def find_phase_times(records, target_label):
    switch_time = None
    transition_start = None
    departure_start = None

    for record in records:
        target = find_target(record, target_label)
        if target is None:
            continue

        time_value = int(record.get("time", 0))
        phase = target.get("intent_phase")
        if transition_start is None and phase == "TurnTransition":
            transition_start = time_value
        if departure_start is None and phase == "FeintDeparture":
            departure_start = time_value
        if switch_time is None and target.get("intent_switch_time") is not None:
            switch_time = float(target["intent_switch_time"])

    if switch_time is None:
        switch_time = transition_start
    return switch_time, transition_start, departure_start


def find_first_crossing_time(times, first_series, second_series):
    differences = first_series - second_series
    crossing_indices = np.where(np.signbit(differences[1:]) != np.signbit(differences[:-1]))[0]
    if crossing_indices.size == 0:
        return None

    index = int(crossing_indices[0])
    x0, x1 = times[index], times[index + 1]
    y0, y1 = differences[index], differences[index + 1]
    if abs(y1 - y0) < 1e-12:
        return float(x1)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def print_threat_rank_table(records, target_label, table_times):
    records_by_time = {
        int(record.get("time", -1)): record
        for record in records
    }

    print(f"\n[PPT table] {target_label} threat score and rank")
    print("Time   | Total score | Rank")
    print("-------+-------------+------")

    for time_value in table_times:
        record = records_by_time.get(int(time_value))
        if record is None:
            print(f"{time_value:>4}s  |     N/A     | N/A")
            continue

        enemies = record.get("enemies", [])
        ordered = sorted(
            enemies,
            key=lambda enemy: float(enemy.get("total_score", 0.0)),
            reverse=True,
        )
        labels = [str(enemy.get("label", "")) for enemy in ordered]
        if target_label not in labels:
            print(f"{time_value:>4}s  |     N/A     | N/A")
            continue

        target = ordered[labels.index(target_label)]
        score = float(target.get("total_score", 0.0))
        rank = labels.index(target_label) + 1
        print(f"{time_value:>4}s  |   {score:.4f}    | {rank}/{len(labels)}")


def export_posterior_figure(args):
    records = load_records(args.data)
    times, source_values, target_values = collect_posterior_series(
        records,
        args.target,
        args.source_intent,
        args.target_intent,
        args.start_time,
        args.end_time,
    )
    switch_time, transition_start, departure_start = find_phase_times(
        records,
        args.target,
    )
    crossing_time = find_first_crossing_time(times, source_values, target_values)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.6, 4.15), dpi=args.dpi)

    if transition_start is not None and departure_start is not None:
        visible_start = max(args.start_time, transition_start)
        visible_end = min(args.end_time, departure_start)
        if visible_start < visible_end:
            ax.axvspan(
                visible_start,
                visible_end,
                color="#dbeafe",
                alpha=0.60,
                linewidth=0,
                label="轨迹转向过渡区",
            )

    ax.plot(
        times,
        source_values,
        color="#d62728",
        linewidth=2.4,
        label=f"P({args.source_intent})",
    )
    ax.plot(
        times,
        target_values,
        color="#1f4e9d",
        linewidth=2.4,
        linestyle="--",
        label=f"P({args.target_intent})",
    )

    if switch_time is not None and args.start_time <= switch_time <= args.end_time:
        ax.axvline(switch_time, color="black", linewidth=1.35, linestyle="--")
        ax.text(
            switch_time,
            0.97,
            f"真实切换时刻\nt = {int(switch_time)} s",
            ha="center",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5},
        )

    if crossing_time is not None:
        crossing_probability = float(np.interp(crossing_time, times, source_values))
        ax.scatter(
            [crossing_time],
            [crossing_probability],
            s=34,
            color="#6b21a8",
            zorder=5,
        )
        ax.annotate(
            f"后验交叉  t≈{crossing_time:.0f}s",
            xy=(crossing_time, crossing_probability),
            xytext=(10, 14),
            textcoords="offset points",
            fontsize=9.5,
            color="#6b21a8",
            arrowprops={"arrowstyle": "->", "color": "#6b21a8", "lw": 1.0},
        )

    ax.annotate(
        f"{source_values[0]:.2f}",
        xy=(times[0], source_values[0]),
        xytext=(5, 8),
        textcoords="offset points",
        color="#d62728",
        fontsize=9.5,
    )
    ax.annotate(
        f"{target_values[0]:.2f}",
        xy=(times[0], target_values[0]),
        xytext=(5, -15),
        textcoords="offset points",
        color="#1f4e9d",
        fontsize=9.5,
    )
    ax.annotate(
        f"{source_values[-1]:.2f}",
        xy=(times[-1], source_values[-1]),
        xytext=(-4, -18),
        textcoords="offset points",
        ha="right",
        color="#d62728",
        fontsize=9.5,
    )
    ax.annotate(
        f"{target_values[-1]:.2f}",
        xy=(times[-1], target_values[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        color="#1f4e9d",
        fontsize=9.5,
    )

    ax.set_title(f"{args.target} 意图后验概率变化", fontsize=15, pad=10)
    ax.set_xlabel("时间 / s", fontsize=12)
    ax.set_ylabel("后验概率", fontsize=12)
    ax.set_xlim(args.start_time, args.end_time)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.23, linewidth=0.7)
    ax.tick_params(labelsize=10.5)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5, ncol=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Intent posterior PNG saved to: {args.output.resolve()}")
    print_threat_rank_table(records, args.target, args.table_times)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the scene-B intent posterior time series as a standalone PNG."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", default="T2")
    parser.add_argument("--source-intent", default="Attack")
    parser.add_argument("--target-intent", default="Feint")
    parser.add_argument("--start-time", type=int, default=75)
    parser.add_argument("--end-time", type=int, default=600)
    parser.add_argument(
        "--table-times",
        type=int,
        nargs="+",
        default=list(DEFAULT_TABLE_TIMES),
        help="Simulation times printed for the PPT threat-score/rank table.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.start_time >= args.end_time:
        parser.error("--start-time must be smaller than --end-time")
    if args.dpi <= 0:
        parser.error("--dpi must be greater than 0")
    return args


if __name__ == "__main__":
    export_posterior_figure(parse_args())
