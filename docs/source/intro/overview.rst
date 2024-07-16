========
Overview
========

Introduction
------------

**Mistify**: Mistify is a library to build neurofuzzy systems and fuzzy neural networks. A neurofuzzy system is a trainable fuzzy system resembling a neural network. Instead of using the matrix multiplication like a neural network it uses the mathematics of fuzzy set theory to transform the inputs of each layer into the outputs of each layer. 

Purpose
-------

To make it easier to build fuzzy systems using PyTorch.

Why Mistify?
-----------

- **Usability**: Mistify is built with PyTorch so the interface is one that is commonly used
- **Completeness**: Mistify contains all of the main components to build a fuzzy system
- **Community-Driven**: 

Design
------

**Mistify** 

Mistify is broken into several modules each 

- **Core**: Functions for executing a vareity of processes in fuzzy set theory.
- **Process**: Preprocessors and postprocessors for the fuzzy system.
- **Fuzzify**: Tools for performing fuzzification and defuzzification 
- **Infer**: Tools for performing fuzzy inference. These are effectively the "hidden layers"
- **Systems**: Fuzzy Systems that make use of Mistify

Getting Started
---------------

Ready to dive in? Check out the `Getting Started`_ guide to set up **Mistify** and run your first experiments.

.. _Getting Started: getting_started.rst
