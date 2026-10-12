# Frozen claim boundary (roadmap Section 1)

## Claims permitted in Article 1

Article 1 may claim that the proposed method:

- estimates airflow in the BMW M30B35 virtual study;
- uses a physics-informed residual architecture;
- adapts a bounded global airflow scale using conditioned lambda and
  injector-command feedback;
- rejects updates that fail operational validity or the frozen scalar
  Fisher-information threshold;
- accepts updates only after a strictly subsequent validation window;
- has a measurable calibration-data requirement under the declared virtual
  mismatch cases; and
- can be represented as a compact software model suitable for later
  embedded deployment.

## Claims excluded from Article 1

The article must not claim that the method:

- is compatible with every automotive engine without engine-specific
  configuration and calibration;
- performs in real time on an STM32H7 or ESP32-S3 before target-hardware
  measurement;
- is physically accepted by an OEM ECU based only on a mathematical voltage
  or frequency curve;
- improves real fuel economy, torque, emissions, knock margin, or
  drivability from simulation alone;
- identifies VE error, MAP gain, lambda bias, injector-flow error, and
  fuel-pressure error independently from lambda alone; or
- replaces all airflow sensing and engine-control functions without a
  reference or feedback assumption.

This file is carried over verbatim from the roadmap and should be checked
against every draft of the manuscript before submission.
