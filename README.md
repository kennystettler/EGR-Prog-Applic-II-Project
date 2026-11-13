This repository contains all of the code pertaining to the EGR Program and Applications II final project.

Group members include Kenneth Stettler, Joshue Heyel, and Caellum Heinlein.

The main goal of this project is to create a working code that of which aims to allow the user to be able to use sensor readings to know how likely a part is to fail 
while also deciding what maintenance is required.

To use the code, simply run at first, then select which of the prompts the user would like to test.

There is also the ability to input data manually to calculate results.

Once the results have finished, there is a visual graph in which the user may use to get a graphical representation of the results.

The expected healthy range for each sensor is estimated from the central 95% of the model's normal training readings and is separate from the configured physical sensor limits. Manual entry displays both ranges; readings outside physical limits require confirmation to analyze, and flagged readings are marked with an asterisk. The all-metrics chart shows the failure estimate, sensor anomalies, and each reading's position relative to its expected healthy range.

The overall health score is based on the weighted mean of each sensor's absolute deviation from its learned normal mean, measured in that sensor's standard deviations. The relative weights for vibration, temperature, current, and runtime are 1.5, 1.5, 1.25, and 0.75. The weighted deviation is mapped to a 0–100 score with `100 / (1 + weighted deviation)`.

Every automated sample, manually entered reading, and requested assessment is appended to `maintenance_history.jsonl` beside the program. Select **8. View audit history** to review timestamps, readings, anomaly scores, failure estimates, and health scores from previous runs. In the range chart, each sensor's healthy interval is centered on the same midpoint and has the same width, so measurements such as runtime remain readable regardless of their physical units or scale.
