# Trend detection using rnn
## Getting Started
This is an project that is used to study how RNN (Recurent Neural Network) works and how to implement RNN from scratch (using numpy). The goal of this project is to build a RNN model that can detect the trend of a given array of number (Increase trend, Decrease trend or Notrend). I supposed that an array has only 1 trend.

## Prerequisites
- Python 3.5+
- Numpy 1.15.2+
- Pandas 0.19.2+
- Sklearn 0.20.0+ (used to print summary report only)
- Pickle 4.0+

## How to run?
- Clone the project
- Run the initscript (init.sh but you have to set the execution right first)
```
chmod u+x init.sh
./init.sh
```
- Create the traing set
```
python3 training/generate.py
```
- Run the training script (main.py)

## Contributing
- quocphan95
