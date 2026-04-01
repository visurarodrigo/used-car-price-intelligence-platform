# Data Dictionary: usedcars.csv

This folder contains the project dataset used for all modeling stages.

- File: `usedcars.csv`
- Row meaning: one row = one car record
- Target column for prediction: `price`

## Column Definitions

1. `symboling`  
   Relative risk/insurance-style index (typically from -3 to +3).

2. `normalized-losses`  
   Relative insurance loss indicator for a vehicle category.

3. `make`  
   Vehicle manufacturer/brand (for example Toyota, Honda, BMW).

4. `aspiration`  
   Engine aspiration type (`std`, `turbo`).

5. `num-of-doors`  
   Number of doors (`two`, `four`).

6. `body-style`  
   Body category (`sedan`, `hatchback`, `wagon`, `hardtop`, `convertible`).

7. `drive-wheels`  
   Drivetrain type (`fwd`, `rwd`, `4wd`).

8. `engine-location`  
   Engine placement (`front`, `rear`).

9. `wheel-base`  
   Distance between front and rear axle centers.

10. `length`  
    Vehicle length feature (normalized/scaled in this dataset).

11. `width`  
    Vehicle width feature (normalized/scaled in this dataset).

12. `height`  
    Vehicle height feature.

13. `curb-weight`  
    Vehicle weight without passengers/cargo.

14. `engine-type`  
    Engine design family (for example `ohc`, `dohc`, `ohcv`, `rotor`).

15. `num-of-cylinders`  
    Cylinder count category (`four`, `six`, `eight`, etc.).

16. `engine-size`  
    Engine size/displacement proxy.

17. `fuel-system`  
    Fuel delivery system (for example `mpfi`, `2bbl`, `1bbl`).

18. `bore`  
    Engine cylinder bore dimension.

19. `stroke`  
    Engine piston stroke dimension.

20. `compression-ratio`  
    Engine compression ratio.

21. `horsepower`  
    Engine power output.

22. `peak-rpm`  
    Engine RPM at peak power.

23. `city-mpg`  
    City fuel economy in miles per gallon.

24. `highway-mpg`  
    Highway fuel economy in miles per gallon.

25. `price`  
    Vehicle selling price (regression target).

26. `city-L/100km`  
    City fuel consumption converted to liters per 100 km.

27. `horsepower-binned`  
    Derived horsepower category (`Low`, `Medium`, `High`).

28. `diesel`  
    Fuel indicator: 1 if diesel, else 0.

29. `gas`  
    Fuel indicator: 1 if gasoline, else 0.

## Notes

- This dataset mixes raw and engineered columns.
- `horsepower-binned`, `city-L/100km`, `diesel`, and `gas` are derived/helper features.
- For model training, treat `price` as target and all other columns as candidate predictors.
