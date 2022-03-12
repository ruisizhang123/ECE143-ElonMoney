### ECE143-ElonMoney

#### Project Summary

Buy stock based on Elon Musk's twitter

* First, use Twitter API to get messages from Musk’s twitter
* Then use sentiment analysis on mentioned companies to create data as input. Use corresponding information from the Stock Dataset as output (labels). Combining both data as our dataset, and split it into training dataset and validation dataset.
* Then, use a logistic regression model to train our model. 
* Last, testing our model in validation dataset to tune hyperparameters.

The real world applications of this solution is that we can make money without spending lots of time looking at others’ twitter all day. We only need to let computers decide which stock should we buy. Besides, the decisions computers make are based on big data. Human cannot view such large data at once. 

#### Get Started

* To run the machine learning model

```bash
python3 ml.py
```

#### Dependencies

* Python >= 3.6
* torch >= 0.4.1
* numpy
* monkeylearn
* pandas
* matplotlib

#### File Structure

* Visualization: ./Data\_visualization/Data\_plot.ipynb
* DataSet: ./data/newstock111.csv
* machine learning module: ./ml.py
* data preprocess: ./write\_crypto.py | ./write\_stock.py

#### Team Member

*  Yonghua Li
*  Hsiu-Wen Yen
*  Yang-Jie Qin
*  Ruisi Zhang
*  Hsiao-Chun Li

