#  neoDL

  This is to run neoantigen intrinsic feature-based deep learning model (`neoDL`) for stratifying IDH wild type GBMs into subgroups with different survival and molecular characteristics

#  Installation

  Python packages “keras”, “pandas” and “lifelines” are in need. Before installing “keras” package, make sure other relevant packages “tensorflow”, “numpy”, “scipy” and “matplotlib” are already installed. If not, run the following commands to install the above python packages respectively. The version of “tensorflow” we used is “1.13.0-rcl”.

    `pip install tensorflow`
    `pip install numpy` 
    `pip install scipy`
    `pip install matplotlib`



######  Run the following command to install “keras”. The version of “keras” we used is “2.2.4”.

    `pip install keras`


######  Run the following command to install “pandas”. The version of “pandas” we used is “0.25.3”.

    `pip install pandas`


######  Run the following command to install “lifelines”. The version of “lifelines” we used is “0.24.3”.

    `pip install lifelines`


#  Running neoDL
    Creat a working directory, e.g.,  ./working
    Put trained model (e.g.,LSTM_model.h5) and neoDL in /working
    Prepare input file in the given format
    Run neoDL:
    `python neoDL.py -o <path> -i <input> -m <model>`
######  Parameters:
    -o:  output directory
    -i:  input file (neoantigen intrinsic features)
    -m:  trained DL model (for IDH wild type GBM, LSTM_model.h5)
    
######  Input file
  Input file is a csv file including sample names(denoted as sample), living days(denoted as days), vital status(denoted as vital, 1 represents for living and 0 represents for dead), mutation numbers(denoted as Num_mutations), neoantigen features and subgroups obtained from k-means(denoted as feat, 1 and 2 represent for two subgroups respectively). 
  
######  Output file
  The output includes cluster results, p-values from survival analysis and survival analysis plots. Cluster results will be written into the input csv file as an independent column, two groups are denoted as 0 and 1 respectively. Those files will be saved into a folder named “test_cluster_result”. P-value from survival analysis will be a separate output csv file, including p-values of all tests. Counts which p-value lower than 0.05 will also be output in the last row. Survival analysis plots will be output as png files, and saved in a folder named “surv_pics”.
    
