 
 #### Requirements:
 - Python 3.7
 - The paths given are absolute paths for my system, provide path to data folder wherever asked for in comments
 
 ### Tasks:
 * Extract 4 different types of time series features from only the CGM data cell array and CGM timestamp cell array.
 * For each time series explain why you chose such feature.
 * Show values of each of the features and argue that your intuition in step b is validated or disproved?
 * Create a feature matrix where each row is a collection of features from each time series. SO if there are 75 time series and your feature length after concatenation of the 4 types of featues is 17 then the feature matrix size will be 75 X 17.
 * Provide this feature matrix to PCA and derive the new feature matrix. Chose the top 5 features and plot them for each time series. 
 * For each feature in the top 5 argue why it is chosen as a top five feature in PCA? (3 points each) total 15.
