from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


AUDIT_LOG_PATH = Path(__file__).with_name("maintenance_history.jsonl")


@dataclass
class SensorReading:
    vibration: float
    temperature: float
    current: float
    runtime_hours: float
    label: int = 0


class SensorDataCleaner:
    """Clean and standardize numerical sensor data."""

    SENSOR_LIMITS = {
        "Vibration": (0.0, 20.0),
        "Temperature": (-40.0, 200.0),
        "Current": (0.0, 100.0),
        "Runtime": (0.0, 200_000.0),
    }

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def clean(self, data: np.ndarray) -> np.ndarray:
        """Convert non-finite sensor values to NaN for model imputation."""
        cleaned = np.asarray(data, dtype=float)

        if cleaned.ndim == 1:
            cleaned = cleaned.reshape(-1, 1)

        cleaned[~np.isfinite(cleaned)] = np.nan
        return cleaned

    def remove_duplicates(self, data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove duplicate rows in original order, preserving their labels."""
        comparable = np.nan_to_num(data, nan=np.finfo(float).max)
        _, unique_indices = np.unique(comparable, axis=0, return_index=True)
        unique_indices.sort()
        return data[unique_indices], labels[unique_indices]

    def validate_reading(self, reading: np.ndarray) -> list[str]:
        """Report sensor readings that are missing or outside configured limits."""
        issues = self.validate_numeric_reading(reading)
        if issues:
            return issues
        return self.sensor_limit_issues(reading)

    def validate_numeric_reading(self, reading: np.ndarray) -> list[str]:
        """Report readings that are missing or not finite."""
        values = np.asarray(reading, dtype=float).reshape(-1)
        if values.size != len(self.SENSOR_LIMITS):
            raise ValueError(f"Expected {len(self.SENSOR_LIMITS)} sensor values.")
        return [
            f"{sensor} reading is missing or not finite."
            for sensor, value in zip(self.SENSOR_LIMITS, values)
            if not np.isfinite(value)
        ]

    def sensor_limit_issues(self, reading: np.ndarray) -> list[str]:
        """Report readings outside configured physical sensor limits."""
        values = np.asarray(reading, dtype=float).reshape(-1)
        if values.size != len(self.SENSOR_LIMITS):
            raise ValueError(f"Expected {len(self.SENSOR_LIMITS)} sensor values.")

        issues = []
        for index, (sensor, (minimum, maximum)) in enumerate(self.SENSOR_LIMITS.items()):
            value = values[index]
            if value < minimum or value > maximum:
                issues.append(
                    f"{sensor} reading {value:g} is outside the configured "
                    f"sensor range ({minimum:g} to {maximum:g})."
                )
        return issues

    def out_of_sensor_range_indices(self, reading: np.ndarray) -> set[int]:
        """Return indices of finite readings outside their configured limits."""
        values = np.asarray(reading, dtype=float).reshape(-1)
        if values.size != len(self.SENSOR_LIMITS):
            raise ValueError(f"Expected {len(self.SENSOR_LIMITS)} sensor values.")
        return {
            index
            for index, (value, (minimum, maximum)) in enumerate(
                zip(values, self.SENSOR_LIMITS.values())
            )
            if np.isfinite(value)
            and (value < minimum or value > maximum)
        }


class PredictiveMaintenanceModel:
    """Train a classifier to predict machine failure risk."""

    FEATURE_NAMES = ("Vibration", "Temperature", "Current", "Runtime")
    HEALTH_WEIGHTS = np.array([1.5, 1.5, 1.25, 0.75])

    def __init__(self):
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )),
            ]
        )
        self.normal_mean: np.ndarray | None = None
        self.normal_std: np.ndarray | None = None
        self.normal_lower: np.ndarray | None = None
        self.normal_upper: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.model.fit(features, labels)
        normal_features = self.model.named_steps["imputer"].transform(features[labels == 0])
        if normal_features.size == 0:
            raise ValueError("Training data must include normal machine readings.")
        self.normal_mean = np.mean(normal_features, axis=0)
        self.normal_std = np.std(normal_features, axis=0)
        self.normal_std[self.normal_std == 0] = 1.0
        self.normal_lower, self.normal_upper = np.percentile(
            normal_features, [2.5, 97.5], axis=0
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def failure_probability(self, features: np.ndarray) -> float:
        """Return the model's estimated probability of failure (class 1)."""
        classifier = self.model.named_steps["classifier"]
        failure_index = np.flatnonzero(classifier.classes_ == 1)
        if failure_index.size == 0:
            raise ValueError("The trained model has no failure class (label 1).")
        return float(self.model.predict_proba(features)[0, failure_index[0]])

    def sensor_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Return each sensor's distance, in standard deviations, from normal."""
        if self.normal_mean is None or self.normal_std is None:
            raise RuntimeError("Train the model before analyzing sensor readings.")
        imputed = self.model.named_steps["imputer"].transform(features)
        return np.abs((imputed[0] - self.normal_mean) / self.normal_std)

    def health_score(self, anomaly_scores: np.ndarray) -> float:
        """Return a weighted 0–100 score from standardized sensor deviations."""
        scores = np.asarray(anomaly_scores, dtype=float).reshape(-1)
        if (
            scores.size != len(self.FEATURE_NAMES)
            or not np.all(np.isfinite(scores))
            or np.any(scores < 0)
        ):
            raise ValueError(
                "Health scoring requires one finite, non-negative anomaly score per sensor."
            )
        weighted_deviation = np.average(scores, weights=self.HEALTH_WEIGHTS)
        return float(100 / (1 + weighted_deviation))

    def expected_range(self, sensor_index: int) -> tuple[float, float]:
        """Return the central 95% range observed for a healthy sensor."""
        if self.normal_lower is None or self.normal_upper is None:
            raise RuntimeError("Train the model before requesting expected ranges.")
        return (
            float(self.normal_lower[sensor_index]),
            float(self.normal_upper[sensor_index]),
        )

    def operating_status(self, reading: np.ndarray, sensor_index: int) -> str:
        """Classify a reading against its learned healthy operating range."""
        value = float(np.asarray(reading, dtype=float).reshape(-1)[sensor_index])
        lower, upper = self.expected_range(sensor_index)
        if value < lower:
            return "low"
        if value > upper:
            return "high"
        return "normal"

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        predictions = self.predict(features)
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        return accuracy, predictions, cm

    def create_assessment_chart(
        self,
        reading: np.ndarray,
        previous_reading: np.ndarray | None = None,
        sample_interval_hours: float = 1.0,
        rapid_change_threshold: float = 2.0,
        outlier_threshold: float = 3.0,
        save_path: str = "failure_assessment.png",
    ) -> tuple[float, str, list[str], float]:
        """Save a chart and return failure probability, strongest anomaly, alerts, and health."""
        reading = np.asarray(reading, dtype=float).reshape(1, -1)
        probability = self.failure_probability(reading)
        anomaly_scores = self.sensor_anomalies(reading)
        strongest_index = int(np.argmax(anomaly_scores))
        strongest_sensor = self.FEATURE_NAMES[strongest_index]
        health = self.health_score(anomaly_scores)

        alerts = []
        for index, score in enumerate(anomaly_scores):
            if score >= outlier_threshold:
                alerts.append(
                    f"{self.FEATURE_NAMES[index]} is an outlier "
                    f"({score:.1f} standard deviations from normal)."
                )

        rapidly_worsening = set()
        if previous_reading is not None:
            if sample_interval_hours <= 0:
                raise ValueError("sample_interval_hours must be greater than zero.")
            previous_reading = np.asarray(previous_reading, dtype=float).reshape(1, -1)
            previous_scores = self.sensor_anomalies(previous_reading)
            for index, (previous_score, current_score) in enumerate(
                zip(previous_scores, anomaly_scores)
            ):
                worsening_rate = (
                    current_score - previous_score
                ) / sample_interval_hours
                if worsening_rate >= rapid_change_threshold:
                    sensor = self.FEATURE_NAMES[index]
                    rapidly_worsening.add(index)
                    alerts.append(
                        f"ALERT: {sensor} is worsening rapidly "
                        f"(anomaly increased by {worsening_rate:.1f} standard "
                        "deviations per hour)."
                    )

        statuses = [
            self.operating_status(reading, index)
            for index in range(len(self.FEATURE_NAMES))
        ]
        for index, status in enumerate(statuses):
            if status != "normal":
                lower, upper = self.expected_range(index)
                direction = "below" if status == "low" else "above"
                alerts.append(
                    f"{self.FEATURE_NAMES[index]} is {direction} its expected "
                    f"healthy range ({lower:g} to {upper:g})."
                )

        colors = [
            "crimson" if index in rapidly_worsening
            else "darkorange" if score >= outlier_threshold
            else "steelblue"
            for index, score in enumerate(anomaly_scores)
        ]
        anomaly_indices = np.minimum(anomaly_scores / 6.0, 1.0) * 100
        chart_threshold = min(outlier_threshold / 6.0, 1.0) * 100

        figure, (probability_axis, sensor_axis, range_axis) = plt.subplots(
            1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [0.8, 1.1, 2.1]}
        )
        probability_axis.barh(["Failure likelihood"], [probability * 100], color="crimson")
        probability_axis.set_xlim(0, 100)
        probability_axis.set_xlabel("Estimated likelihood (%)")
        probability_axis.set_title("Overall model estimate")
        probability_axis.text(
            min(probability * 100 + 2, 80), 0, f"{probability:.1%}", va="center"
        )

        sensor_axis.barh(self.FEATURE_NAMES, anomaly_indices, color=colors)
        sensor_axis.axvline(
            chart_threshold, color="darkorange", linestyle="--", label="Outlier threshold"
        )
        sensor_axis.set_xlim(0, 100)
        sensor_axis.set_xlabel("Relative sensor anomaly index (not probability, %)")
        sensor_axis.set_title(f"Sensor anomaly index — strongest: {strongest_sensor}")
        sensor_axis.legend(handles=[
            Patch(color="crimson", label="Worsening rapidly"),
            Patch(color="darkorange", label="Outlier"),
            Patch(color="steelblue", label="Within range"),
            Line2D([0], [0], color="darkorange", linestyle="--", label="Outlier threshold"),
        ])

        sensor_limits = list(SensorDataCleaner.SENSOR_LIMITS.values())
        out_of_range = {
            index
            for index, value in enumerate(reading.reshape(-1))
            if value < sensor_limits[index][0] or value > sensor_limits[index][1]
        }
        for index, (value, status) in enumerate(zip(reading.reshape(-1), statuses)):
            lower, upper = self.expected_range(index)
            midpoint = (lower + upper) / 2
            half_width = (upper - lower) / 2
            if half_width == 0:
                half_width = self.normal_std[index]
            position = (value - midpoint) / half_width
            visible_position = np.clip(position, -2, 2)
            marker_color = {
                "low": "gold",
                "high": "crimson",
                "normal": "forestgreen",
            }[status]
            range_axis.barh(index, 4, left=-2, color="gainsboro", height=0.55)
            range_axis.barh(index, 2, left=-1, color="palegreen", height=0.55)
            range_axis.scatter(
                visible_position,
                index,
                color=marker_color,
                marker=">" if position > 2 else "<" if position < -2 else "o",
                s=55,
                zorder=3,
            )
            marker = index in out_of_range or status != "normal"
            marker_label = f"{value:g}{'*' if marker else ''}"
            range_axis.annotate(
                marker_label,
                (visible_position, index),
                xytext=(5 if visible_position < 1.7 else -5, 8),
                textcoords="offset points",
                ha="left" if visible_position < 1.7 else "right",
                fontsize=8,
            )

        range_axis.set_yticks(range(len(self.FEATURE_NAMES)))
        range_axis.set_yticklabels(self.FEATURE_NAMES)
        range_axis.invert_yaxis()
        range_axis.set_xlim(-2, 2)
        range_axis.set_xticks(
            [-2, -1, 0, 1, 2],
            ["-2x", "low edge", "midpoint", "high edge", "+2x"],
        )
        range_axis.set_xlabel(
            "Distance from each healthy midpoint (healthy range = -1x to +1x)"
        )
        range_axis.set_title("Centered healthy ranges (* = flagged)")
        range_axis.legend(handles=[
            Patch(color="gainsboro", label="Display span (±2 healthy half-ranges)"),
            Patch(color="palegreen", label="Expected healthy range"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gold", label="Too low"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", label="Too high"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="forestgreen", label="In range"),
        ], fontsize=8)
        figure.suptitle("Predictive Maintenance Assessment")
        figure.tight_layout()
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(figure)

        return probability, strongest_sensor, alerts, health


def append_audit_record(
    model: PredictiveMaintenanceModel,
    reading: np.ndarray,
    source: str,
    event: str,
    alerts: list[str] | None = None,
    log_path: Path = AUDIT_LOG_PATH,
) -> None:
    """Persist a diagnostic snapshot as one JSON Lines audit record."""
    values = np.asarray(reading, dtype=float).reshape(-1)
    if values.size != len(model.FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("Audit logging requires one finite reading per sensor.")
    anomaly_scores = model.sensor_anomalies(values.reshape(1, -1))
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "event": event,
        "sensor_readings": dict(zip(model.FEATURE_NAMES, values.tolist())),
        "anomaly_scores_standard_deviations": dict(
            zip(model.FEATURE_NAMES, anomaly_scores.tolist())
        ),
        "failure_probability": model.failure_probability(values.reshape(1, -1)),
        "overall_health": model.health_score(anomaly_scores),
        "alerts": alerts or [],
    }
    try:
        with log_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, allow_nan=False) + "\n")
    except OSError as error:
        raise OSError(
            f"Unable to append the audit record to {log_path}: {error}"
        ) from error


def read_audit_history(log_path: Path = AUDIT_LOG_PATH) -> list[dict]:
    """Load persisted diagnostic records in chronological order."""
    try:
        with log_path.open("r", encoding="utf-8") as audit_file:
            return [json.loads(line) for line in audit_file if line.strip()]
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unable to read audit history from {log_path}: {error}"
        ) from error


def print_audit_history(log_path: Path = AUDIT_LOG_PATH) -> None:
    """Display persisted diagnostic readings and assessment results."""
    records = read_audit_history(log_path)
    if not records:
        print(f"No audit history found at {log_path}.")
        return
    print(f"\nAudit history ({log_path}):")
    for record in records:
        readings = ", ".join(
            f"{name}={value:g}" for name, value in record["sensor_readings"].items()
        )
        anomalies = ", ".join(
            f"{name}={score:.2f}"
            for name, score in record["anomaly_scores_standard_deviations"].items()
        )
        print(
            f"\n{record['timestamp_utc']} | {record['source']} | {record['event']}\n"
            f"  Readings: {readings}\n"
            f"  Anomaly scores (standard deviations): {anomalies}\n"
            f"  Failure estimate: {record['failure_probability']:.1%} | "
            f"Health: {record['overall_health']:.1f}/100"
        )


def generate_sensor_data(num_samples: int = 500, failure_rate: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic sensor data for healthy and failing machines."""
    rng = np.random.default_rng(42)
    num_failures = int(num_samples * failure_rate)
    num_normal = num_samples - num_failures

    normal_vibration = rng.normal(loc=1.2, scale=0.4, size=num_normal)
    normal_temperature = rng.normal(loc=70, scale=10, size=num_normal)
    normal_current = rng.normal(loc=15, scale=3, size=num_normal)
    normal_runtime = rng.uniform(50, 500, size=num_normal)

    failure_vibration = rng.normal(loc=4.5, scale=1.0, size=num_failures)
    failure_temperature = rng.normal(loc=110, scale=14, size=num_failures)
    failure_current = rng.normal(loc=28, scale=5, size=num_failures)
    failure_runtime = rng.uniform(150, 800, size=num_failures)

    features = np.column_stack([
        np.concatenate([normal_vibration, failure_vibration]),
        np.concatenate([normal_temperature, failure_temperature]),
        np.concatenate([normal_current, failure_current]),
        np.concatenate([normal_runtime, failure_runtime]),
    ])

    labels = np.concatenate([
        np.zeros(num_normal, dtype=int),
        np.ones(num_failures, dtype=int),
    ])

    # Add a few missing values to mimic realistic machine data
    missing_mask = rng.random(features.shape) < 0.03
    features[missing_mask] = np.nan

    return features, labels


def print_sensor_reading(
    model: PredictiveMaintenanceModel,
    cleaner: SensorDataCleaner,
    reading: np.ndarray,
    previous_reading: np.ndarray,
    sensor_index: int,
    sample_interval_hours: float,
) -> None:
    """Display one sensor's value, anomaly level, and trend alert."""
    issues = cleaner.validate_numeric_reading(reading)
    if issues:
        print("Cannot assess this reading:")
        for issue in issues:
            print(f"- {issue}")
        return

    anomaly_scores = model.sensor_anomalies(reading.reshape(1, -1))
    score = anomaly_scores[sensor_index]
    name = model.FEATURE_NAMES[sensor_index]
    value = reading[sensor_index]
    lower, upper = model.expected_range(sensor_index)
    physical_issue = sensor_index in cleaner.out_of_sensor_range_indices(reading)
    status = model.operating_status(reading, sensor_index)
    issue = physical_issue or status != "normal"
    print(f"{name}: {value:g}{'*' if issue else ''}")
    print(f"Expected healthy range: {lower:g} to {upper:g}")
    if physical_issue:
        limit_issues = cleaner.sensor_limit_issues(reading)
        matching_issue = next(
            (item for item in limit_issues if item.startswith(f"{name} reading")), None
        )
        print(f"ALERT: {matching_issue} This reading is likely invalid.")

    if status == "low":
        print(f"ALERT: {name} is below its expected healthy range.")
    elif status == "high":
        print(f"ALERT: {name} is above its expected healthy range.")
    else:
        print(f"{name} is within its expected healthy range.")
    print(f"Deviation from normal: {score:.2f} standard deviations")

    if score >= 3.0:
        print(f"ALERT: {name} is an outlier and needs inspection.")
    else:
        print(f"{name} is not currently an outlier.")

    if not cleaner.validate_numeric_reading(previous_reading):
        previous_score = model.sensor_anomalies(previous_reading.reshape(1, -1))[sensor_index]
        worsening_rate = (score - previous_score) / sample_interval_hours
        if worsening_rate >= 2.0:
            print(
                f"ALERT: {name} is worsening rapidly "
                f"({worsening_rate:.2f} standard deviations per hour)."
            )


def print_reading_ranges(
    model: PredictiveMaintenanceModel,
    cleaner: SensorDataCleaner,
    reading: np.ndarray,
) -> None:
    """Display all current readings alongside expected and physical ranges."""
    physical_issues = cleaner.out_of_sensor_range_indices(reading)
    for index, name in enumerate(model.FEATURE_NAMES):
        value = reading[index]
        lower, upper = model.expected_range(index)
        status = model.operating_status(reading, index)
        status_text = {
            "low": "TOO LOW",
            "high": "TOO HIGH",
            "normal": "within expected range",
        }[status]
        marker = "*" if index in physical_issues or status != "normal" else ""
        print(
            f"{name}: {value:g}{marker} "
            f"(expected {lower:g} to {upper:g}) - {status_text}"
        )

    for issue in cleaner.sensor_limit_issues(reading):
        print(f"ALERT: {issue} This reading is likely invalid.")
    if physical_issues or any(
        model.operating_status(reading, index) != "normal"
        for index in range(len(model.FEATURE_NAMES))
    ):
        print("* indicates a reading outside its expected healthy range or physical limits.")


def interactive_menu(
    model: PredictiveMaintenanceModel,
    cleaner: SensorDataCleaner,
    reading: np.ndarray,
    previous_reading: np.ndarray,
) -> None:
    """Let the user inspect a sensor, overall health, all metrics, or new readings."""
    sample_interval_hours = 1.0
    reading_source = "automated"
    while True:
        print("\nChoose what to inspect:")
        for index, name in enumerate(model.FEATURE_NAMES, start=1):
            lower, upper = model.expected_range(index - 1)
            status = model.operating_status(reading, index - 1)
            physical_issue = index - 1 in cleaner.out_of_sensor_range_indices(reading)
            marker = "*" if physical_issue or status != "normal" else ""
            print(
                f"{index}. {name}: {reading[index - 1]:g}{marker} "
                f"(expected {lower:g} to {upper:g})"
            )
        print("5. Overall engine health")
        print("6. All metrics and failure chart")
        print("7. Enter new sensor readings")
        print("8. View audit history")
        print("0. Exit")

        choice = input("Selection: ").strip()
        if choice == "0":
            return
        if choice in {"1", "2", "3", "4"}:
            sensor_index = int(choice) - 1
            print_sensor_reading(
                model,
                cleaner,
                reading,
                previous_reading,
                sensor_index,
                sample_interval_hours,
            )
            append_audit_record(
                model,
                reading,
                reading_source,
                f"sensor_inspection:{model.FEATURE_NAMES[sensor_index]}",
            )
        elif choice == "5":
            issues = cleaner.validate_numeric_reading(reading)
            if issues:
                for issue in issues:
                    print(issue)
                continue
            print_reading_ranges(model, cleaner, reading)
            anomalies = model.sensor_anomalies(reading.reshape(1, -1))
            health = model.health_score(anomalies)
            print(f"Overall engine health: {health:.1f}/100")
            print(
                f"Formula: 100 / (1 + weighted mean standardized deviation); "
                f"weighted deviation is "
                f"{np.average(anomalies, weights=model.HEALTH_WEIGHTS):.2f}."
            )
            append_audit_record(model, reading, reading_source, "overall_health")
        elif choice == "6":
            issues = cleaner.validate_numeric_reading(reading)
            previous_issues = cleaner.validate_numeric_reading(previous_reading)
            if issues or previous_issues:
                for issue in issues + previous_issues:
                    print(issue)
                continue
            print("\nCurrent sensor readings:")
            print_reading_ranges(model, cleaner, reading)
            probability, strongest_sensor, alerts, health = model.create_assessment_chart(
                reading,
                previous_reading=previous_reading,
                sample_interval_hours=sample_interval_hours,
            )
            print(f"Overall failure likelihood: {probability:.1%}")
            print(f"Overall engine health: {health:.1f}/100")
            print(
                f"Strongest sensor anomaly: {strongest_sensor} "
                "(an indicator, not a confirmed mechanical root cause)."
            )
            anomaly_scores = model.sensor_anomalies(reading.reshape(1, -1))
            likely_issue_indices = {
                index
                for index, score in enumerate(anomaly_scores)
                if score >= 3.0
                or model.operating_status(reading, index) != "normal"
            }
            likely_issue_indices.update(
                index
                for index, name in enumerate(model.FEATURE_NAMES)
                if any(
                    alert.startswith(f"ALERT: {name} is worsening rapidly")
                    for alert in alerts
                )
            )
            if likely_issue_indices:
                likely_cause = max(
                    likely_issue_indices, key=lambda index: anomaly_scores[index]
                )
                print(
                    f"Most likely contributor to the detected issue: "
                    f"{model.FEATURE_NAMES[likely_cause]}."
                )
            if alerts:
                for alert in alerts:
                    print(alert if alert.startswith("ALERT:") else f"ALERT: {alert}")
            else:
                print("No sensor outliers or rapid deterioration detected.")
            print("Chart saved as failure_assessment.png")
            append_audit_record(
                model,
                reading,
                reading_source,
                "all_metrics_assessment",
                alerts,
            )
        elif choice == "7":
            print("Enter current readings. Each shows its expected and physical ranges:")
            new_values = []
            for index, name in enumerate(model.FEATURE_NAMES):
                expected_lower, expected_upper = model.expected_range(index)
                physical_lower, physical_upper = cleaner.SENSOR_LIMITS[name]
                while True:
                    try:
                        value = float(input(
                            f"{name} (expected {expected_lower:g} to "
                            f"{expected_upper:g}; sensor limits {physical_lower:g} "
                            f"to {physical_upper:g}): "
                        ))
                    except ValueError:
                        print("Enter a numeric sensor value.")
                        continue

                    if not np.isfinite(value):
                        print("Enter a finite sensor value.")
                        continue

                    if value < physical_lower or value > physical_upper:
                        print(
                            f"ALERT: {value:g} is outside the configured "
                            f"{name.lower()} sensor limits and is likely impossible."
                        )
                        proceed = input(
                            "Run anyway with this value? Enter y to proceed; "
                            "otherwise enter a new value: "
                        ).strip().lower()
                        if proceed not in {"y", "yes"}:
                            continue
                        print(f"{name} accepted with a warning; it will be marked *.")
                    new_values.append(value)
                    break

            new_reading = np.asarray(new_values, dtype=float)
            while True:
                try:
                    interval = float(input("Hours since the previous reading: "))
                except ValueError:
                    print("Enter a numeric time interval.")
                    continue
                if not np.isfinite(interval) or interval <= 0:
                    print("The time interval must be a finite number greater than zero.")
                    continue
                break

            previous_reading = reading
            reading = new_reading
            sample_interval_hours = interval
            reading_source = "user-entered"
            print(f"Reading updated. Trend interval: {interval:g} hour(s).")
            append_audit_record(
                model,
                reading,
                reading_source,
                "manual_reading_entered",
            )
        elif choice == "8":
            print_audit_history()
        else:
            print("Choose one of the listed menu options.")


def main() -> None:
    features, labels = generate_sensor_data()
    cleaner = SensorDataCleaner()
    features = cleaner.clean(features)
    features, labels = cleaner.remove_duplicates(features, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = PredictiveMaintenanceModel()
    model.fit(X_train, y_train)
    accuracy, predictions, confusion = model.evaluate(X_test, y_test)

    print("Predictive Maintenance Model (synthetic demonstration data)")
    print("=" * 58)
    print(f"Test accuracy on synthetic data: {accuracy:.2%}")
    print("Confusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=["Normal", "Failure Imminent"]
    ))
    print("Sensor range limits are configurable demo limits in SensorDataCleaner.")
    print("Connect real sensor data and set limits for the actual hardware before use.")

    current_reading = np.array([4.8, 118.0, 30.0, 620.0])
    previous_reading = np.array([2.8, 95.0, 22.0, 600.0])
    append_audit_record(
        model,
        current_reading,
        "automated",
        "automated_reading",
    )
    interactive_menu(model, cleaner, current_reading, previous_reading)


if __name__ == "__main__":
    main()
