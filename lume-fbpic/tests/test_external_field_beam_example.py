"""The external-field example, `docs/examples/external_field_beam.py`, on a shortened run."""

from __future__ import annotations

import pytest

from short_runs import on_cpu


def test_the_corrector_deflects_the_beam_as_predicted(tmp_path):
    import external_field_beam

    model = on_cpu(
        external_field_beam.build_model(str(tmp_path), simulation_length=1.0e-4)
    )
    model.set({"corrector_by": 2.0})

    assert model.simulator.external_fields[0].field_functions["By"] == 2.0
    assert external_field_beam.deflection_rad(model) == pytest.approx(
        external_field_beam.predicted_deflection_rad(model), rel=0.03
    )


def test_the_example_has_a_corrector_and_beam_actions():
    import external_field_beam

    model = external_field_beam.build_model()
    variables = model.supported_variables

    assert variables["corrector_by"].unit == "T"
    assert {"beam_gamma", "beam_sigma_r", "beam_charge"} <= set(variables)
