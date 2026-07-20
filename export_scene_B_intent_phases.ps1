param(
    [string]$OutputRoot = "results_fig",
    [int]$MaxFramesPerPhase = 25,
    [int]$Fps = 8,
    [switch]$SkipAssessment
)

$ErrorActionPreference = "Stop"

if (-not $SkipAssessment) {
    python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py `
        --experiment scene_B_deception `
        --output-root $OutputRoot `
        --visual-step 1
}

$sceneDir = Join-Path $OutputRoot "scene_B_deception"
$dataPath = Join-Path $sceneDir "visual_data.json"
if (-not (Test-Path -LiteralPath $dataPath)) {
    throw "Missing $dataPath. Run this script without -SkipAssessment first."
}

$segments = @(
    @{ Name = "T2_attack_approach.gif"; Start = 0; End = 125 },
    @{ Name = "T2_turn_transition.gif"; Start = 126; End = 225 },
    @{ Name = "T2_feint_departure.gif"; Start = 226; End = 600 }
)

foreach ($segment in $segments) {
    $outputPath = Join-Path $sceneDir $segment.Name
    python .\visualizer_3d_matplotlib_export_animation.py `
        --data $dataPath `
        --format gif `
        --start-time $segment.Start `
        --end-time $segment.End `
        --max-frames $MaxFramesPerPhase `
        --fps $Fps `
        --output $outputPath
}

$posteriorOutput = Join-Path $sceneDir "T2_intent_posterior.png"
python .\export_scene_B_intent_posterior.py `
    --data $dataPath `
    --target T2 `
    --start-time 75 `
    --end-time 600 `
    --output $posteriorOutput

python .\export_scene_B_phase_keyframes.py `
    --data $dataPath `
    --output-dir $sceneDir `
    --dpi 300

Write-Host "Scene B phase GIFs, posterior PNG, and phase-final PNGs saved under $sceneDir."
