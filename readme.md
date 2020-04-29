Dataset from: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

Download it and unzip it as modelnet40_normal_resampled

Run dataset.py to do the preprocessing.

Then run: train.py 

You can run it for 5 epochs to make it faster: train.py --epochs 5

Then you can test on the testing set with: test.py

Also, to get plots after training you can run: plotter.py
Note: plotter.py is hardcoded to only run with the default model save location, "saved_models".