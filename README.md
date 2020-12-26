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
    Run neoDL

