Predictive Maintenance for Motors or Machines Use sensor readings such as vibration, temperature, current, and runtime to predict whether a machine is operating normally or may fail soon. Uses: NumPy, scikit-learn, classes, data cleaning.

Generate a chart displaying the visual likelihood of failure across all monitored metrics. The menu should show the number that the sensor is reading, along with the range in which it should expect to be in.There should be a visual that shows in which where the parts should be. This would have a lower and a higher range in which when the parts fall out of that range, can be demonstrated by a visual (red too high, yellow too low).

Highlight outliers to pinpoint the most probable root cause of potential failure.

Component-Specific Failure Alerts

Detect rapidly failing specific components or aspects in real time.

Send targeted alerts for the specific failing component rather than a generic system failure notification.

Check for outliers as long as getting rid of duplicates.

Check for impossible reading that are outside of the range of the sensors.

Add in a final overall health score of the engine by taking the average standard deviation of the engine parts and using that to determine overall health.

Add in a menu that can help choose which reading they would like, along with an option for an overall health reading described above. Also allow the user to enter their own sensor data, but when the user is entering data, make sure there is a range in which the user should expect the sensor to be reading.

If the user gives a number that is impossible to reach, give an error message in which tells the user it is likely an impossible number, but ask if they would like to run anyway, or enter a new number.

If a check is ran that scans everything, prompt the user with information on which item is the reason for the faulure (if any). 