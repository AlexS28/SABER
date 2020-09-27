# SABR
SABR: Synchronous and Autonomous Bots with Reinforcement Learning

Instructions: 

Running the mapping and data collection files (at the moment requires python2 with ros)

First, several variables need to be manually-modified. Go to the gazebo_map.launch file, and ensure the
correct world#.world is referenced. Then go to save_map.sh file, and input the desired name of map# that will be saved
in the maps/ folder. Go to gazebo_collect.launch file, and ensure the world#.world and the map# are referenced
correctly. Now go to play1.sh and run it in a terminal, wait until everything is open. Go to play2.py in a separate
terminal and run the file - this file will start the mapping process. Once this file is done running, run the 
play3.py file (at the end of the code in this file, you can set the name of the dataset that will be saved in
the data_collection folder - it must be of the form dataset#.csv). 

Running the RNN (at the moment requires python3):

The only option that needs to be changed in the rnn.py file, is whether the datasets collected in the dataset 
collection folder are based on lidar scans or rgbd images. You can have any number of datasets in that folder,
as long as all datasets are using either only lidar scans or only rgbd images. The output of the rnn.py is a 
rnn model that is saved in the rnn_models folder. 

