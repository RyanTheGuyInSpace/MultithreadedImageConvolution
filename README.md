# Multithreaded Image Convolution
> [!NOTE]
> Have PyCharm installed. In File -> Settings, ensure that numpy, imageio, joblib, IPython and matplotlib are installed in your Python environment. Open the project folder in PyCharm. Right click on single.py or multi.py and click "Run...". If all dependencies are installed properly, the above output will be produced.

> [!WARNING]
> gpu.py is still a work in progress
## Single Threaded Implementation
File Name: `single.py`
### Running Instructions:
Have Python installed on your system as well as pip.
Make sure to `pip install numpy imageio joblib IPython matplotlib`
Run `py single.py`
### Expected Output:
```
(601, 900)
(601, 900)
(601, 900)
ELAPSED_TIME: 8.462000846862793  seconds
Output image size is  (601, 900, 3)
```
## Multithreaded Implementation
File Name: `multi.py`
### Running instructions:
Have Python installed on your system as well as pip.
Make sure to `pip install numpy imageio joblib IPython matplotlib`
Run `py multi.py`
### Expected Output:
```
Convolution complete
Convolution complete
Convolution complete
[[[82 82 84]
  [78 78 80]
  [79 79 81]
  ...
  [68 68 68]
  [ 0  0  0]
  [ 0  0  0]]

 [[85 85 87]
  [82 82 84]
  [78 78 80]
  ...
  [72 72 72]
  [ 0  0  0]
  [ 0  0  0]]

 [[83 83 85]
  [79 79 81]
  [75 75 77]
  ...
  [68 70 69]
  [ 0  0  0]
  [ 0  0  0]]

 ...

 [[67 65 60]
  [53 51 60]
  [51 49 54]
  ...
  [25 31 40]
  [ 0  0  0]
  [ 0  0  0]]

 [[ 0  0  0]
  [ 0  0  0]
  [ 0  0  0]
  ...
  [ 0  0  0]
  [ 0  0  0]
  [ 0  0  0]]

 [[ 0  0  0]
  [ 0  0  0]
  [ 0  0  0]
  ...
  [ 0  0  0]
  [ 0  0  0]
  [ 0  0  0]]]
ELAPSED_TIME: 3.4749996662139893  seconds
```
