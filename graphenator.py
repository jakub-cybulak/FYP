# -*- coding: utf-8 -*-

## Header =====================================================================

'''
Create plots from data outputs of "multi_resnet.py".

Two graphs are produced. One is the "epoch" graph showing training and 
validation loss and metric data. One is the "test" graph showing predicted
and experimental data.

Inputs:
    
metrics_fp - CSV file of four float columns (row order = epoch)
test_fp - CSV file of two float columns (row order is test,pass,section,
          test...)

See "USER TO EDIT" section for more details.
'''

## USER TO EDIT - START =======================================================

# Normaliser (When wear was normalised 0-1, this is the value to undo that)
Norm = 1.76

# Full filepath to metrics output file
metrics_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
              r"\02_Experimental\09_Metrics\Val_ResNet50v2_DeepDrop2_1.csv")

# Data labels and styles corresponding to column number
labels = ['Training MAE','Validation MAE','Training RMSE','Validation RMSE']
styles = ['k-','k--','m-.','m:']
# Plot settings
topx = 60
topy = 0.4
intervalx = 5
intervaly = 0.1

# Full filepath to test results output file
test_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
           r"\02_Experimental\09_Metrics\Tes_ResNet50v2_DeepDrop2_1.csv")
# r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
# r"\02_Experimental\09_Metrics\ResNet50v2_ConicalDeep_1.txt"

# Locations at which data should be split (final of previous), add None 
# as first and last
locs = [None, 891, None]

# Rolling average window width
avg_width = 50

# Data labels and styles corresponding to column number
test_labels = ['Predicted','Predicted Running Average','Experimental']
test_styles = ['k.','b--','r-']
# Plot settings
test_topxs = [900,1400]
test_topys = [1.3,1.3]
test_botxs = [0,0]
test_botys = [0,0]
test_intervalxs = [100,200]
test_intervalys = [0.1,0.1]

## USER TO EDIT - END =========================================================

## Libraries ==================================================================

# Python version: 3.9.13
# Spyder version: 5.2.2

# Import libraries - 3rd party
import numpy as np # 1.21.5
import pandas as pd # 1.4.4
import matplotlib.pyplot as plt # 3.5.2

## Main: Epoch Graph ==========================================================

# Load metrics and un-normalise
metrics_matrix = np.loadtxt(metrics_fp,'float','#',',') * Norm
n_rows, n_cols = np.shape(metrics_matrix) # Dimmensions

# Set up the plot
fig, ax = plt.subplots()
ax.set_ylim([0, topy])
ax.set_xlim([0, topx])
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Wear /mm')
plt.yticks(np.arange(0, topy + 0.01, intervaly))
plt.xticks(np.arange(0, topx + 1, intervalx))
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Plot all of the metrics
for n_col in range(n_cols):
    ax.plot(range(1,n_rows+1), metrics_matrix[:,n_col], styles[n_col],
            label=labels[n_col])

# Show the legend (without title)
ax.legend(title=None)

# Show the plot
plt.show()

## Main: Test Graph ===========================================================

# Get test results as panda df
test_df = pd.read_csv(test_fp, header=None)

# Split into sub df's and un-normalise
predictions = test_df[0].astype(float) * Norm
actuals = test_df[1].astype(float) * Norm

# Plot all of the metrics
for idx in range(len(locs)-1):
    # Extract individual test and calculate rolling average
    curr_pred = predictions.iloc[locs[idx]:locs[idx+1]]
    curr_roll = curr_pred.rolling(avg_width,min_periods=1).mean()
    curr_actu = actuals.iloc[locs[idx]:locs[idx+1]]
    # Package 
    metrics_matrix = [curr_pred,curr_roll,curr_actu]
    
    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_ylim([test_botys[idx], test_topys[idx]])
    ax.set_xlim([test_botxs[idx], test_topxs[idx]])
    ax.set_xlabel('Snip Number')
    ax.set_ylabel('Wear /mm')
    plt.yticks(np.arange(test_botys[idx], test_topys[idx] + 0.01,
                         test_intervalys[idx]))
    plt.xticks(np.arange(test_botxs[idx], test_topxs[idx] + 0.01,
                         test_intervalxs[idx]))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Plot all of the metrics
    for n_col in range(len(locs)):
        ax.plot(range(len(metrics_matrix[0])), metrics_matrix[n_col],
                test_styles[n_col], label=test_labels[n_col],markersize=10,
                markerfacecolor=(0, 1, 0, 0.2))
    
    # Show the legend (without title)
    ax.legend(title=None)
    
    # Show the plot
    plt.show()
    
    # Calculations...
    
    e = curr_pred - curr_actu # Error
    ae = e.abs() # Absolute Error
    mae = ae.mean(None) # Mean Absolute Error
    
    se = ae * ae # Squared Error
    mse = se.mean(None) # Mean Squared Error
    rmse = mse ** (1/2) # Root Mean Squared Error
    
    # Rolling Average - \/\/\/
    ra_e = curr_roll - curr_actu # Error
    ra_ae = ra_e.abs() # Absolute Error
    ra_mae = ra_ae.mean(None) # Mean Absolute Error
    
    ra_se = ra_ae * ra_ae # Squared Error
    ra_mse = ra_se.mean(None) # Mean Squared Error
    ra_rmse = ra_mse ** (1/2) # Root Mean Squared Error
    
    # Print Calculations
    print("Graph: " + str(idx + 1) + "; MAE: " + str(mae) +
          "; RMSE: " + str(rmse))
    print("Graph: " + str(idx + 1) + "; RA-MAE: " + str(ra_mae) +
          "; RA-RMSE: " + str(ra_rmse))