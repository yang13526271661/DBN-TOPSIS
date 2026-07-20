param(
    [int]$GifStep = 25,
    [int]$GifFps = 6,
    [string]$OutputRoot = "results_fig"
)

$ErrorActionPreference = "Stop"

$experiments = @(
    "typical_fixed",
    "scene_A_saturation",
    "scene_B_deception",
    "scene_C_border",
    "dynamic_formation",
    "dynamic_heterogeneous",
    "altitude_missile",
    "member_degradation"
)

foreach ($exp in $experiments) {
    Write-Host ""
    Write-Host "========== Running experiment: $exp =========="
    python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py --experiment $exp --output-root $OutputRoot

    Write-Host "========== Rendering GIF: $exp =========="
    python .\visualizer_3d_matplotlib_export_animation.py --scenario $exp --step $GifStep --fps $GifFps --format gif
}

Write-Host ""
Write-Host "All experiment data and GIF files have been generated under $OutputRoot."
