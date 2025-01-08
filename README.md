# Cu(I)-Catalyzed Regiodivergent Borylation of Alkynes
Code and results of the machine learning workflow. The results shown in the paper "_Cu(I)-Catalyzed Regiodivergent Borylation of Alkynes: Artificial Intelligence Accelerated Ligands Design for Regio-selection and Reaction Optimization_" can be replicated with the material found here.

# Initial setup
We'll show an example of how to execute the main script on a new environment using conda, but it can also be used without virtual environments and only using pip. First, we'll create the environment:

```console
conda create -n boryl python=3.9
conda activate boryl
```

To execute the main script, we need to install our [mlworkflow](https://github.com/aitenea/ml-workflow) package that encapsulates all the steps from training models to performing predictions with them:

```console
pip install mlworkflow@git+https://github.com/aitenea/ml-workflow
```
