Predictive Maintenance for Motors or Machines Use sensor readings such as vibration, temperature, current, and runtime to predict whether a machine is operating normally or may fail soon. Uses: NumPy, scikit-learn, classes, data cleaning.

Generate a chart displaying the visual likelihood of failure across all monitored metrics.

Highlight outliers to pinpoint the most probable root cause of potential failure.

Component-Specific Failure Alerts

Detect rapidly failing specific components or aspects in real time.

Send targeted alerts for the specific failing component rather than a generic system failure notification.

Check for outliers as long as getting rid of duplicates.

Check for impossible reading that are outside of the range of the sensors.

Add in a final overall health score of the engine by taking the average standard deviation of the engine parts and using that to determine overall health.

Add in a menu that can help choose which reading they would like, along with an option for an overall health reading described above.