# SCADA sensor documentation and data quality caveats

This retrieval corpus accompanies the turbine status assistant. These are operational caveats, not claims that a sensor is faulty on every reading.

## General SCADA readings
- Sensor columns named `sensor_N` are anonymized in this dataset. Their physical meaning and engineering units are not supplied; do not infer a component or unit from the column number.
- The source contains ten-minute aggregates and summary columns such as `_avg`, `_min`, `_max`, and `_std`. Aggregates can hide short-lived excursions between samples.
- Zero values may be real, a stopped-state value, a communication default, or a missing-value encoding. Check operating state and neighboring measurements before treating zero as a confirmed fault.
- Flat or repeated readings can indicate stable operation or a frozen/stale channel. Compare timestamps, related channels, and turbine state.
- Farm exports do not guarantee identical sensor availability or calibration. Compare within the same turbine/farm where possible; raw magnitudes across farms may not be directly comparable.
- Event windows come from maintenance/event logs and provide weak labels. Start/end times may be approximate, event descriptions are free text, and an event interval is not a point-by-point diagnosis.
- Missing timestamps, irregular gaps, and status transitions can disrupt sequence interpretation. Predictions should be qualified when recent coverage is sparse or the turbine is stopped.

## Wind and power channels
- Wind speed and power depend on operating mode, curtailment, and environmental conditions. Low power alone is not proof of a turbine fault.
- Power, reactive power, and wind channels may use different scales or aggregation conventions. Check the feature name and context before comparing values.

## Hydraulic, temperature, and bearing event notes
- Event descriptions mention hydraulic oil temperature, oil level/leakage, and bearing damage in some cases. Treat these as historical annotations, not validated sensor-to-fault mappings.
- Temperature spikes can be affected by sensor placement, warm-up, load, or stale values; confirm persistence and corroborating channels before escalating.
