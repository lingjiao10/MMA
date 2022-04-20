# Modeling Missing Annotations for Incremental Learning in Object Detection

## Official PyTorch implementatation based on Detetron - Fabio Cermelli, Antonino Geraci, Dario Fontanel, Barbara Caputo

Despite the recent advances in the field of object detection, common architectures are still ill-suited to incrementally detect new categories over time. They are vulnerable to catastrophic forgetting: they forget what has been already learned while updating their parameters in absence of the original training data. Previous works extended standard classification methods in the object detection task, mainly adopting the knowledge distillation framework. However, we argue that object detection introduces an additional problem, which has been overlooked. While objects belonging to new classes are learned thanks to their annotations, if no supervision is provided for other objects that may still be present in the input, the model learns to associate them to background regions. We propose to handle these missing annotations by revisiting the standard knowledge distillation framework. Our approach outperforms current state-of-the-art methods in every setting of the Pascal-VOC dataset. We further propose an extension to instance segmentation, outperforming the other baselines. In this work, we propose to handle the missing annotations by revisiting the standard knowledge distillation framework. We show that our approach outperforms current state-of-the-art methods in every setting of the Pascal-VOC 2007 dataset. Moreover, we propose a simple extension to instance segmentation, showing that it outperforms the other baselines.

# Code will be released soon!