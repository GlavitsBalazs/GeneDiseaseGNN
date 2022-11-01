# GeneDiseaseGNN
Predicting gene-disease associations using heterogeneous graph neural networks

Knowledge of gene-disease associations (GDAs) is helpful in further understanding the mechanisms that underly diseases and syndromes. This understanding can, in turn, aid in developing novel methods for diagnosis, prevention, or treatment. However, the picture is far from complete. Supposedly, the majority of true GDAs are yet undiscovered. The fact that experimental verification of new hypothetical GDAs is a resource and time-intensive task hinders discovery. Perhaps a way of predicting GDAs that likely exist could assist in reducing resources expended on experiments with a negative outcome.

The size of genomic measurement results, along with the collective body of knowledge amassed by the entire field of medicine, has far outgrown the scale which is easily understandable by the individual. For this reason, machine learning-based methods have become central in recent biomedical analyses. One such method is the graph neural network. This model can learn from examples and so predict whether or not relevant relationships exist between pairs of entities.

In my work, I've integrated various biomedical databases to form a heterogeneous knowledge graph, where the sets of genes and diseases are two distinct types. I then used this data to train and evaluate graph neural networks with different architectures for the task of GDA prediction.
