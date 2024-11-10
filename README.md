
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
this uses a simple relu activation function, but will change this as we experiment

## dropout layer
this has a single dropout layer set to 0.1 probability, applied after layer 1.
this will essentially drop a neuron during training 10% of the time, to prevent the model overfitting.
it helps the model not rely on particular neuron configurations
dropout is not applied during inference