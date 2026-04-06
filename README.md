# Haul-mark
1st Phase
We tested tree-based models (like LightGBM/XGBoost). We played around with features (mean speed, altitude gain, etc.) and found that total distance travelled, total idle hours, and so on were of good importance.
We took the sum of decreases in fuel as our target variable until the new data was released. Our error was very high, and then we realised that there were so many pairs of consecutive points where the latter had lower fuel. Even a small error in each fuel_volume would be multiplied by a large number because we were aggregating all such decreases. We then shifted to using the RFID dataset that we hadn't used previously (using the formula initial_level − final_level + refuel_volume). Even a large error like 1 L would be insignificant compared to the previous approach.

2nd Phase
With tree-based models and similar features, we achieved a score of around 6000. To improve upon this, we tried encoding the route that the dumper takes using KMeans clustering of the starting and ending latitude and longitude of a shift. We also dropped some bad training rows, fixed the fuel calculation slightly, removed low-importance features, and added new ones. These were:

Gross elevation gain/loss: We included this feature because intuitively the dumper consumes more fuel uphill and less fuel downhill.
Number of stops per shift and stop density: Since stop-start cycles significantly increase fuel consumption, we believed this was a good feature to include (density is stops/hr).
Mean fuel consumed by a particular dumper per shift over all training data: We added this because it better represents the dumper than simply label-encoding the vehicle name. For example, a dumper may be allocated to more difficult tasks, increasing wear and thus fuel consumption — basically, the higher the mean, the more the wear, and the more the fuel consumed.


3rd Phase
We tried using analog_input_1 (the dump switch) to improve the score from around 600 (still using tree-based models ;-;). We plotted graphs for analog_input_1 and found that a threshold of 2–2.5 was good enough to determine that the dumper was dumping (analog_input_1 >= 2 ⟹ dumping). We took the rows where the dump switch signal was reliable, then trained a classifier to predict whether the dumper was dumping at each timestamp using window-based features. However, making predictions was very slow as the model had to infer dumping status for millions of rows. So we found dumping-site centroids using KMeans and restricted inference to rows within a certain distance threshold (~100–150 m) of the nearest dump site centre. We then added features like number of dump cycles, dumps per hour, and so on, but there was little to no improvement.
We then attempted to use .gpkg files (GeoPackage zone files containing bench and stockpile polygons) to directly identify dump locations and assign dump-cycle labels, but found that the score was not improving. We concluded that the probable cause was that the actual dump locations drift slightly day by day, making the static zone boundaries an unreliable signal.
We tried replacing the number of dump cycles with the frequency of the dumper slowing down (with low or zero speed) for 3–5 minutes within a shift, since dumper location drift has a negligible effect on this — but this did not work either (don't know why ;-;). Finally, we computed the "Economy Speed" (the speed at which mean km/L of fuel consumed is maximum), which came out to around 30 km/h. We added a small feature for the amount of time the dumper spends inside the economy speed range (taken as 20–40 km/h) and outside it. (That didn't work either ;-;)

Final Model
The final model is a LightGBM regressor stacked with a Ridge meta-learner, trained on per-shift features engineered from raw telemetry.

The Features Considered were :
Gradient features: percent uphill/downhill, total climb work (altitude gain × distance), climb work per km, mean uphill grade, max grade.
Heading/turn features: heading std, mean heading change, sharp/harsh turn counts and per-km rates — proxy for road difficulty and driving style.
Dump-cycle features (revamped): Dropped both analog_input_1 and static .gpkg zones . Built a pure GPS-based pipeline instead: extract stop events from speed → compute approach/departure speed asymmetry per stop (loaded truck approaches slow, departs fast after tipping → positive asymmetry = dump signal) → cluster stops per (vehicle, date) scoring on visit frequency, duration consistency, and speed asymmetry to pick the dump cluster → propagate labels back to raw rows. Per-shift aggregates: n_dumps, dump duration stats, CV of dump duration, dumps/hr, dump work score, mean approach/departure speeds.
Vehicle lag features (biggest single win): prev_fuel, 3-shift and 7-shift rolling mean, 3-shift rolling std. A vehicle's fuel consumption is very stable across shifts — knowing the last shift's consumption reduces uncertainty massively. Test lags are computed from the last N training shifts, so no leakage.
Target encoding: vehicle, route cluster, and operator mode smoothed-mean encoded against fuel_consumed.

Training
Time-based 2-fold CV (month 1 → 2, months 1–2 → 3) with early stopping . The mean best iteration across folds was used to retrain the final model on all data. Compared to earlier phases, regularisation was tightened: num_leaves 63 → 47, stronger L1/L2, lower feature_fraction/bagging_fraction, added min_split_gain.
Stacking and Final Prediction
A Ridge meta-learner was trained on OOF LightGBM predictions + squared predictions + three anchor features (prev_fuel, rolling3_fuel, vehicle_fuel_mean), giving 0.3–0.5 RMSE gain. Final predictions = 80% full-data LGB + 20% Ridge correction.
