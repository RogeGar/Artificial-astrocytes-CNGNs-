The NeuronGlia unit can, in theory, work for any CNN. To introduce the unit to a CNN architecture, use the function "transform_architecture" of the NeuronGlia script.

Example:

model = torchvision.models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

NeuronGliaUnit.transform_architecture(model, Astro_str, Astro_weak)

The Astro_str and Astro_weak are hyperparameters of the artificial astrocyte. We recommend using values between [0.5, 1.5] for both.
