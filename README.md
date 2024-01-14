# Mistify

Mistify is a library built on PyTorch for building Fuzzy Neural Networks or Neurofuzzy Systems. 

## Installation

```bash
pip install mistify
```

## Brief Overview

Mistify consists of several subpackages

- **mistify.fuzzify**: Modules for building fuzzifiers and defuzzifiers.
- **mistify.infer**: Modules for performing inference operations such as Or Neurons, Intersections, etc.
- **mistify.process**: Modules for preprocessing or postprocessing on the data to input into the fuzzy system
- **mistify.activate**: Activations to perform on memberships or to use before fuzzification etc.
- **mistify.functional**: Contains the functional definitions for many of the operations.

## Usage

Mistify's primary prupose is to build neurofuzzy systems. 

Here is a (non-working) example that uses alternating Or and And neurons.
```bash

class FuzzySystem(nn.Module):

    def __init__(
        self, in_features: int, h1: int, h2: int, out_features: int
    ):
        self.fuzzifier = SigmoidFuzzifier(in_features, categories)
        self.flatten = FlattenCat()
        self.layer1 = OrNeuron(in_features * categories, h1)
        self.layer2 = AndNeuron(h1, h2)
        self.layer3 = OrNeuron(h2, out_features * out_categories)
        self.deflatten = DeflattenCat(out_categories)
        self.defuzzifier = SigmoidDefuzzifier(
            out_features, out_categories)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        m = self.fuzzifier(x)
        m = self.flatten(m)
        m = self.layer1(m)
        m = self.layer2(m)
        m = self.layer3(m)
        m = self.deflatten(m)
        return self.defuzzifier(m)

```

Since it uses Torch, these fuzzy systems can easily be stacked. 


## Contributing

To contribute to the project

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Citing this Software

If you use this software in your research, we request you cite it. We have provided a `CITATION.cff` file in the root of the repository. Here is an example of how you might use it in BibTeX:
