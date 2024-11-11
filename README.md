# Introduction
This is my very simple example binary classifier.

The model will take in array of features (sized at 10 features), and then will predict if the sum of the first half of the features is larger than the sum of the second half of the features.  

if the sum of the first half of the features is greater than the sum of the second half of the features, it will be labeled "1", otherwise will be labelled "0"

## Data Generation
the following will generate data for the simple binary classifier

```bash
python data_generator.py --num_samples 2000  --input_size 10
```

## Train the model

```bash
python train.py
```

## Evaluate the model

```bash
python evaluate.py
```

## activation function
this uses a silu activation function.  started with ReLU but switched to SiLU for better results

## dropout layer
this has a dropout layer set to 0.1 probability, applied after each hidden layer.
this will essentially drop a neuron during training 10% of the time, to prevent the model overfitting.
it helps the model not rely on particular neuron configurations
dropout is not applied during inference