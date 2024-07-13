We provide all the scripts required to train the CNNs using the proposed artificial astrocyte.

The NeuronGlia unit can, in theory, work for any CNN. To introduce the unit to a CNN architecture, use the function "transform_architecture" of the NeuronGlia script.

Example:

model = torchvision.models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

NeuronGliaUnit.transform_architecture(model, Astro_str, Astro_weak)

The Astro_str and Astro_weak are hyperparameters of the artificial astrocyte. We recommend using values between [0.5, 1.5] for both.

Also, the script named ICML2024_final_trainer automates the training of the architectures we used in our experiments and saves the activations in the same layer that we use for the similarity analysis using this script. The instructions to use the ICML2024_final_trainer are:

1.- THE CODE IS GOING TO ASK FOR A DIRECTORY, THAT IS THE ADDRESS WHERE YOU HAVE THE  FOLDER CONTAINING THE DATASET. THE DATASET ONLY HAS TO HAVE THE IMAGES IN FOLDERS WITH THE NAME OF THEIR RESPECTIVE CLASSES.
            
 2.- THEN, THE CODE IS GOING TO ASK FOR THE NAME OF THE DATASET, THAT IS THE NAME OF THE FOLDER WHERE THE DATA IS.

3.- THE CODE IS GOING TO ASK FOR A MODEL TO TRAIN. PLEASE, PROVIDE ONE OF THE FOLLOWING:
        
        EfficientNet-V2-Small, RegNet-Y-400MF, RegNet-X-400MF,
        Shufflenet-v2-x1_0, Shufflenet-v2-x1_5, Shufflenet-v2-x2_0
        
4.- NEXT, ENTER THE DIRECTORY WHERE YOU WANT TO SAVE THE RESULTS.
    
5.- FINALLY, PROVIDE THE NUMBER OF TRAINING EPOCHS AND THE ASTROCYTES HYPERPARAMETERS WE RECOMMEND VALUES BETWEEN [0.5, 1.5] FOR THE "Str" ADN "Weak" ASTROCYTE'S HYPERPARAMETERS.

THE SCRIPT SAVES THE ACTIVATIONS OF THE LAST CONVOLUTIONAL LAYER IN A CSV FILE IN CASE YOU WANT TO
APLY A SIMILARITY ANALYSIS ON THEM.

IT ALSO SAVES THE MODEL'S PARAMETERS IN A STATE_DICT AND THE CLASSIFICATION PERFORMANCE METRICS
AT THE END OF TRAINING.