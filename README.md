# Mistify

Mistify is a library built on PyTorch for building Neurofuzzy Systems. A neurofuzzy system is a trainable fuzzy system, typically consisting of a fuzzifier, rules layers, and a defuzzifier. Mistify provides a variety of modules to use at each layer of the pipeline 

## Installation

```bash
pip install mistify
```

## Brief Overview

Mistify consists of subpackages for inference operations, fuzzification and defuzzification, and preprocessing and postprocessing. It incldes 

- **mistify**: The core functions used for fuzzification and inference.
- **mistify.fuzzify**: Modules for building fuzzifiers and defuzzifiers. Has a variety of shapes or other fuzzification and defuzzification modules to use.
- **mistify.infer**: Modules for performing inference operations such as Or Neurons, Intersections, Activations etc.
- **mistify.process**: Modules for preprocessing or postprocessing on the data to input into the fuzzy system
- **mistify.systems**: Modules for building systems more easily.
- **mistify.utils**: Utilities used by other modules in Mistify. 

## Usage

Mistify's primary prupose is to build neurofuzzy systems or fuzzy neural networks using the the framework of PyTorch. 

Here is a (non-working) example that uses alternating Or and And neurons.
```bash

class FuzzySystem(nn.Module):

    def __init__(
        self, in_features: int, h1: int, h2: int, out_features: int
    ):

        # Use for these builders for buliding a neuron
        # In this case, tehre is no wait fou
        AndNeruon = BuildAnd().no_wf().inter_on().prob_union()
        OrNeuron = BuildOr().no_wf().union_on().prob_inter()

        # 
        self.fuzzifier = mistify.fuzzify.SigmoidFuzzifier.from_linspace(
            n_terms, 'min_core', 'average'
        )
        self.flatten = FlattenCat()
        self.layer1 = OrNeuron(in_features * categories, h1)
        self.layer2 = AndNeruon(h1, h2)
        self.layer3 = OrNeuron(h2, out_features * out_categories)
        self.deflatten = DeflattenCat(out_categories)

        self.defuzzifier = mistify.fuzzify.IsoscelesFuzzyConverter.from_linspace(
            out_terms, 'min_core', 'average'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        m = self.fuzzifier(x)
        m = self.flatten(m)
        m = self.layer1(m)
        m = self.layer2(m)
        m = self.layer3(m)
        # use to prepare for defuzzification
        m = self.deflatten(m)
        return self.defuzzifier.defuzzify(m)

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
