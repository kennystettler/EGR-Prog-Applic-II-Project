import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        values = np.asarray(reading, dtype=float).reshape(-1)
        if values.size != len(self.SENSOR_LIMITS):
            raise ValueError(f"Expected {len(self.SENSOR_LIMITS)} sensor values.")

        issues = []
        for index, (sensor, (minimum, maximum)) in enumerate(self.SENSOR_LIMITS.items()):
            value = values[index]
            if not np.isfinite(value):
                issues.append(f"{sensor} reading is missing or not finite.")
            elif value < minimum or value > maximum:
                issues.append(
                    f"{sensor} reading {value:g} is outside the configured "
                    f"sensor range ({minimum:g} to {maximum:g})."
                )
        return issues


class PredictiveMaintenanceModel:
    """Train a classifier to predict machine failure risk."""

    FEATURE_NAMES = ("Vibration", "Temperature", "Current", "Runtime")

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

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.model.fit(features, labels)
        normal_features = self.model.named_steps["imputer"].transform(features[labels == 0])
        if normal_features.size == 0:
            raise ValueError("Training data must include normal machine readings.")
        self.normal_mean = np.mean(normal_features, axis=0)
        self.normal_std = np.std(normal_features, axis=0)
        self.normal_std[self.normal_std == 0] = 1.0

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
        """Map the average absolute sensor deviation to a simple 0–100 score."""
        return float(100 / (1 + np.mean(anomaly_scores)))

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

        colors = [
            "crimson" if index in rapidly_worsening
            else "darkorange" if score >= outlier_threshold
            else "steelblue"
            for index, score in enumerate(anomaly_scores)
        ]
        anomaly_indices = np.minimum(anomaly_scores / 6.0, 1.0) * 100
        chart_threshold = min(outlier_threshold / 6.0, 1.0) * 100

        figure, (probability_axis, sensor_axis) = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.5]}
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
        figure.suptitle("Predictive Maintenance Assessment")
        figure.tight_layout()
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(figure)

        return probability, strongest_sensor, alerts, health


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
    issues = cleaner.validate_reading(reading)
    if issues:
        print("Cannot assess this reading:")
        for issue in issues:
            print(f"- ALERT: {issue}")
        return

    anomaly_scores = model.sensor_anomalies(reading.reshape(1, -1))
    score = anomaly_scores[sensor_index]
    name = model.FEATURE_NAMES[sensor_index]
    value = reading[sensor_index]
    print(f"{name}: {value:g}")
    print(f"Deviation from normal: {score:.2f} standard deviations")

    if score >= 3.0:
        print(f"ALERT: {name} is an outlier and needs inspection.")
    else:
        print(f"{name} is not currently an outlier.")

    if not cleaner.validate_reading(previous_reading):
        previous_score = model.sensor_anomalies(previous_reading.reshape(1, -1))[sensor_index]
        worsening_rate = (score - previous_score) / sample_interval_hours
        if worsening_rate >= 2.0:
            print(
                f"ALERT: {name} is worsening rapidly "
                f"({worsening_rate:.2f} standard deviations per hour)."
            )


def interactive_menu(
    model: PredictiveMaintenanceModel,
    cleaner: SensorDataCleaner,
    reading: np.ndarray,
    previous_reading: np.ndarray,
) -> None:
    """Let the user inspect a sensor, overall health, all metrics, or new readings."""
    sample_interval_hours = 1.0
    while True:
        print("\nChoose what to inspect:")
        for index, name in enumerate(model.FEATURE_NAMES, start=1):
            print(f"{index}. {name}")
        print("5. Overall engine health")
        print("6. All metrics and failure chart")
        print("7. Enter new sensor readings")
        print("0. Exit")

        choice = input("Selection: ").strip()
        if choice == "0":
            return
        if choice in {"1", "2", "3", "4"}:
            print_sensor_reading(
                model,
                cleaner,
                reading,
                previous_reading,
                int(choice) - 1,
                sample_interval_hours,
            )
        elif choice == "5":
            issues = cleaner.validate_reading(reading)
            if issues:
                for issue in issues:
                    print(f"ALERT: {issue}")
                continue
            anomalies = model.sensor_anomalies(reading.reshape(1, -1))
            health = model.health_score(anomalies)
            print(f"Overall engine health: {health:.1f}/100")
            print(
                f"Formula: 100 / (1 + average sensor deviation); "
                f"average deviation is {np.mean(anomalies):.2f} standard deviations."
            )
        elif choice == "6":
            issues = cleaner.validate_reading(reading)
            previous_issues = cleaner.validate_reading(previous_reading)
            if issues or previous_issues:
                for issue in issues + previous_issues:
                    print(f"ALERT: {issue}")
                continue
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
            if alerts:
                for alert in alerts:
                    print(alert if alert.startswith("ALERT:") else f"ALERT: {alert}")
            else:
                print("No sensor outliers or rapid deterioration detected.")
            print("Chart saved as failure_assessment.png")
        elif choice == "7":
            print("Enter current readings:")
            try:
                new_reading = np.array([
                    float(input(f"{name}: "))
                    for name in model.FEATURE_NAMES
                ])
                interval = float(input("Hours since the previous reading: "))
            except ValueError:
                print("Enter numeric sensor values and a numeric interval.")
                continue
            issues = cleaner.validate_reading(new_reading)
            if issues:
                for issue in issues:
                    print(f"ALERT: {issue}")
                print("The reading was not applied. Check the sensor limits.")
                continue
            if interval <= 0:
                print("The time interval must be greater than zero.")
                continue
            previous_reading = reading
            reading = new_reading
            sample_interval_hours = interval
            print(f"Reading updated. Trend interval: {interval:g} hour(s).")
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
    interactive_menu(model, cleaner, current_reading, previous_reading)


if __name__ == "__main__":
    main()
