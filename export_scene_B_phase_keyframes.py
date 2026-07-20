import argparse
from pathlib import Path

from visualizer_3d_matplotlib_export_animation import (
    compute_global_axis_limits,
    export_scene_png,
    freeze_enemy_display_types,
    load_data,
)


KEYFRAMES = (
    (125, "T2_attack_approach_last_125s.png"),
    (225, "T2_turn_transition_last_225s.png"),
    (600, "T2_feint_departure_last_600s.png"),
)


def find_frame_index(records, requested_time):
    exact = [
        idx
        for idx, record in enumerate(records)
        if int(record.get("time", -1)) == requested_time
    ]
    if not exact:
        raise ValueError(f"No visualization record exists at t={requested_time}s")
    return exact[0]


def main():
    parser = argparse.ArgumentParser(
        description="Export the final scene-only PNG for each scene-B intent phase."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results_fig/scene_B_deception/visual_data.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_fig/scene_B_deception"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be greater than 0")

    _, records = load_data(args.data)
    freeze_enemy_display_types(records)
    fixed_limits = compute_global_axis_limits(records)

    for time_value, filename in KEYFRAMES:
        frame_idx = find_frame_index(records, time_value)
        export_scene_png(
            records,
            args.output_dir / filename,
            frame_idx,
            dpi=args.dpi,
            fixed_limits=fixed_limits,
        )


if __name__ == "__main__":
    main()
