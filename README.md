# ToNNi 
### Fully connected neural network

ToNNi is a python library used for training and testing neural networks

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tonni.

```bash
pip install toNNi
```

## Usage

```python
import toNNi

# returns dataset
X_train, Y_train, X_test, Y_test = toNNi.Dataloader.loadMNIST(50000,10000)

# define architecture
network = [

      Dense(256),
      Batchnorm(),
      ReLU(),
      Dense(128),
      Batchnorm(),
      ReLU(),
      Dense(10),
      Softmax(),

]

# define model
loss = CCE()
optimizer = "adam"
model = Model(
      network=network,
      loss=loss,
      optimizer=optimizer,
      lr=0.0001,
      epochs=2000050,
      batch_size=64
      )

#start training
model.fit(
      X_train,
      Y_train,
      X_test,
      Y_test,
      diagnostics=True,
      shuffle=True,
      name="snek",
      validate=True
      )
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## TODO

### 

* Loss function
    * ~~CCE~~
    * ~~BCE~~
    * R2
* Layers
    * ~~Conv~~
    * ~~Maxpool~~
    * ~~Dropout~~
* Optimizers
    * ~~Abram~~
    * ~~SHG~~
* Plotting
    * ~~Conv outputs~~
    * ~~Gradients~~
    * ~~Architecture~~
* misc
    * ~~Optimization~~
    * ~~Refacture 2.0~~
    * Launch module
    * Website stats
    * ...
    * ~~weight initialization~~
    * ~~batch shuffle optimization~~




