# SyntheticMedData

## Environment

- python3.x
- Anaconda (Recommended)

## building rotMNIST dataset

```
python build_rotMNIST.py
```

## Output

```
rotMNIST
  |
  |- data/*.npy : numpy data (Tx784(=28x28))
  |- rotMNIST.json: data and label ID list
  |- rotMNIST.label.json: description for rach label ID
  |- rotMNIST.png
```
