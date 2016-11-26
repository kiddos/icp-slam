Grid base SLAM
==============

Simulation of grid base SLAM (Simultaneous Localization and Mapping) implemented using Iterative Closest Points

## Dependencies
* opencv
* sklearn
* numpy
* scipy
* matplotlib

## ICP pseudo code
```
best_error = max_scan_range * 2
repeat until convergence
  find the correspond points using KDTree
  find the translation T and rotation R using singular value decomposition
  transform B using T and R
  compute error with least mean square

  if error < best_error 
    best_error = error
  else
    break
end
```
