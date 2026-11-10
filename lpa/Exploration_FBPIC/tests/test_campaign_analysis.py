from pathlib import Path

import pandas as pd

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.campaign_analysis import (
    analyze_campaign,
    plot_charge_slice,
    plot_feature_slice,
    plot_feasibility_slice,
    plot_input_pca_projections,
)


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_campaign_analysis_uses_scalar_store_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    config = campaign.config.__class__(
        name=campaign.config.name,
        template_script=campaign.config.template_script,
        template_class=campaign.config.template_class,
        run_root=tmp_path / "runs",
        parameters=campaign.config.parameters,
        stages=campaign.config.stages,
        analysis=campaign.config.analysis,
        execution=campaign.config.execution,
    )
    rows = []
    for number in range(18):
        parameters = config.complete_parameters("stage_1")
        for offset, name in enumerate(
            config.stage_settings("stage_1").active_parameters
        ):
            parameter = config.parameters[name]
            parameters[name] = (
                float(parameter.lower)
                + (float(parameter.upper) - float(parameter.lower))
                * ((number + offset) % 18)
                / 17
            )
        rows.append(
            {
                "run_id": f"sim_{number:06d}",
                "stage": "stage_1",
                "campaign_fingerprint": config.fingerprint(),
                "status": "ok",
                **{f"p:{name}": value for name, value in parameters.items()},
                **{
                    f"f:{name}": number + index
                    for index, name in enumerate(config.exploration_features)
                },
            }
        )
    config.run_root.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(
        config.run_root / "campaign_runs.parquet", index=False
    )

    result = analyze_campaign(
        Campaign(config),
        "stage_1",
        tmp_path / "analysis",
        folds=3,
        data_path=config.run_root / "campaign_runs.parquet",
        vector_sim=6,
        input_pca_x=3,
        input_pca_y=4,
    )

    assert result.report["complete_training_rows"] == 18
    assert result.report_path.is_file()
    assert result.plot_path.is_file()
    assert result.report["cross_validation"]["folds"] == 3

    charge_slice = plot_charge_slice(
        Campaign(config),
        "stage_1",
        tmp_path / "analysis" / "charge-slice.png",
        data_path=config.run_root / "campaign_runs.parquet",
        grid_size=12,
    )

    assert charge_slice.is_file()

    uz_slice = plot_feature_slice(
        Campaign(config),
        "stage_1",
        tmp_path / "analysis" / "uz-slice.png",
        "longitudinal_mean_uz_02",
        data_path=config.run_root / "campaign_runs.parquet",
        grid_size=12,
    )

    assert uz_slice.is_file()

    data = pd.read_parquet(config.run_root / "campaign_runs.parquet")
    data.loc[:5, "status"] = "error"
    data.loc[:5, "message"] = (
        "No scalar diagnostic results found in results/datapoint.json"
    )
    data.to_parquet(config.run_root / "campaign_runs.parquet", index=False)
    no_beam_projection = analyze_campaign(
        Campaign(config),
        "stage_1",
        tmp_path / "analysis-no-beam",
        folds=3,
        data_path=config.run_root / "campaign_runs.parquet",
    )
    assert no_beam_projection.plot_path.is_file()
    feasibility_slice = plot_feasibility_slice(
        Campaign(config),
        "stage_1",
        tmp_path / "analysis" / "feasibility-slice.png",
        data_path=config.run_root / "campaign_runs.parquet",
        grid_size=12,
    )

    assert feasibility_slice.is_file()

    projections = plot_input_pca_projections(
        Campaign(config),
        "stage_1",
        tmp_path / "analysis" / "input-projections",
        data_path=config.run_root / "campaign_runs.parquet",
    )

    assert len(projections) == 4
    assert all(path.is_file() for path in projections)
