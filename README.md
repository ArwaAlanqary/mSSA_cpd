## Data Format
All time series data are stored in headerless and indexless csv files inside the folder "data"


## Change point detector output
Each algorithm has two outputs the first is a list of tupules of the form (start, end), and the second is the score given by thr algorihtm. If no score is avalibale the score output is None