
# Musculoskeletal X-Ray Classification

## Description
The aim of this assignment was to build a model that could classify 
images of musculoskeletal X-Rays into two predefined categories: 
*abnormal* and *normal*. Two different architectures were trained 
and tested using the whole dataset, and then a third more complex 
architecture was created using only the body parts that produced 
the lower scores, in an effort to improve their predictions. 
In particular, a CNN was trained from scratch and a pre-trained 
DenseNet169 model was fine-tuned with the whole dataset, whereas 
an ensemble of a pre-trained DenseNet169 and a pre-trained 
InceptionV3 was fine-tuned using only the poor-performance body 
parts. The Cohen Kappa score was used as the main metric (as
suggested by the competition), which measures the inter-rater 
agreement for categorical items.

## Dataset
The dataset that was used is MURA 
(https://stanfordmlgroup.github.io/competitions/mura/). Mura is
a large dataset of musculoskeletal radiographs where each study 
is manually labeled by radiologists as either normal or abnormal.
The dataset contains X-Rays from seven different body parts: 
wrists, forearms, humeri, hands, shoulders, elbows and fingers

[Training Instances](https://github.com/StMarou/Musculoskeletal-X-Ray-Classification/blob/master/static/training_instances.png)

## Preprocessing
First, since the shape of the X-Rays was variable, all the X-Rays were 
resized to have the same shape (224x224x3 for the CNN and 
DenseNet169 models and 299x299x3 for the InceptionV3 model). Then,
the pixel values were divided by 255 to convert them to range from 0 
to 1. A validation set (15% of the original training set) was used 
to decide when the training will be terminated. Data augmentation 
was also applied to (only) the training set, by randomly flipping 
the images horizontally and rotating them up to 30 degrees. Since 
the whole dataset could not fit in memory, `keras`’ 
`ImageDataGenerator` was used to apply these transformations. Also,
due to resource limitations, `ImageDataGenerator` proved to be 
extremely useful, since instead of creating new images which would 
slow down the training process, it uses random transformations of 
the original X-Rays in each epoch, while keeping the number of 
training instances the same.

## Model Architectures
### CNN
The first architecture that was built and tested consists of 4 
stacked CNN layers. The first CNN layer has a kernel size of 5x5, 
whereas the rest have a kernel size of 3x3. Images were also padded 
and ReLU was used as the activation function. The number of 
filters for the first CNN layer was equal to 128 and was gradually 
decreased as the model became deeper. Between the CNN layers, 
batch normalization and max pooling was applied, reducing the 
image dimension in half. A dropout rate of 0.35 was also applied 
between some of the layers. Lastly, a fully connected layer was 
added on top, with 64 neurons and ReLU as an activation, which 
produced the last set of features before the output layer 
(1 neuron with sigmoid). Adam was used as the optimizer with a 
learning rate of 1e-4. To avoid overfitting, early stopping was 
used with a patience of 10 epochs. Furthermore, using `keras`’ 
`ReduceLROnPlataeu`, if the validation loss did not decrease for 
5 consecutive epochs, the learning rate was multiplied by 0.5.

### DenseNet169
The second architecture that was tried was a pre-trained 
DenseNet169 [1] model. The DenseNet architecture, instead of 
creating representation features from extremely deep architectures, 
it reuses features that were produced from every layer. Each layer 
has access to all the preceding feature-maps in its block and, 
therefore to the, as G. Huang, et. al. (2017) call it in their 
paper, network’s “collective knowledge” . This has two benefits; 
first, it combats the problem of vanishing gradients, as the 
information has a better flow among all layers of the network, and 
secondly, it requires fewer parameters than the traditional CNN 
architectures, as it does not learn redundant feature maps.

[DenseNet architecture](https://github.com/StMarou/Musculoskeletal-X-Ray-Classification/blob/master/static/densenet.png)

For our problem, a pre-trained (on `ImageNet`) DenseNet169 model 
was fine-tuned. The top of the model was removed, and a global 
average pooling layer was added before the output layer to flatten 
the previous input. The output layer contained one neuron and a 
sigmoid as the activation function. The network was initially 
trained with frozen weights on the DenseNet169 architecture, with 
Adam as the optimizer and a learning rate of 5e-3. The callbacks 
that were used were the same as the previous model. The training 
lasted 10 epochs, and then the whole network was retrained 
(without frozen weights), with a learning rate of 1e-5 
(`ReduceLROnPlataeu`’s and `EarlyStopping`’s patience was 
decreased to 3 and 6 respectively, due to resource limitations).

## Results
The DenseNet169 model significantly outperformed the CNN model. 
The cohen kappa score was 75% larger for the DenseNet model, 
and its accuracy was larger by 0.12. The rest of the metrics 
were also significantly higher.

|        Metric        | CNN  | DenseNet169 |
|:--------------------:|:----:|:----:|
| Cohen Kappa          | 0.31 | 0.55 |
| Accuracy             | 0.66 | 0.78 |
| Recall (normal)      | 0.81 | 0.91 |
| Recall (abnormal)    | 0.50 | 0.63 |
| Precision (normal)   | 0.64 | 0.73 |
| Precision (abnormal) | 0.71 | 0.87 |
| F1 (normal)          | 0.71 | 0.81 |
| F1 (abnormal)        | 0.71 | 0.73 |

Both models seem to mostly misclassify abnormal X-Rays, which 
led to low recall (for the abnormal class), F1 (for the abnormal 
class) and precision (for the normal class) scores, although to 
a lesser extent for the DenseNet169 model. This might have been 
caused due to the imbalance of the training set.

However, the performance was not the same for all body parts. 
X-Rays of humeri, followed by X-Rays of wrists and elbows had the 
three highest cohen kappa scores (above 0.6). On the contrary, 
X-Rays of hands, followed by X-Rays of shoulders and fingers had 
the three lowest (under 0.5) cohen kappa scores (among the 
DenseNet169’s prediction). 

[Kappas](https://github.com/StMarou/Musculoskeletal-X-Ray-Classification/blob/master/static/kappas_bodyparts.png)

## Attempt to improve results: Ensemble
Since the resources were limited, only the three poorest performing 
body parts were used, in an attempt to improve their results. In 
particular, using only X-Rays of hands, shoulders and fingers, an 
ensemble of a pre-trained DenseNet169 and a pre-trained 
InceptionV3 [2] was fine-tuned.

The idea of the original Inception model [3] was that instead of 
using very deep networks, multiple filters of different sizes were 
used on the same level, resulting in parallel layers. The 
InceptionV3 is just a more advanced and optimized version of the 
original model. Having a total of 42 layers, it introduced four 
major modifications. The first one was the factorization into 
smaller convolutions. On top of the 1x1 convolutions that reduced 
the resulting dimension, which were generously used in the first 
version, in the V3 version larger convolutions were factorized 
into smaller ones. For example, a 5x5 convolution would be 
replaced with 2 stacked 3x3 convolutions which is computationally 
cheaper. However, the factorization did not stop there; spatial 
factorization into asymmetric convolutions was also introduced. For 
example, a 3x3 convolution would be replaced with a 1x3 followed 
by a 3x1 convolution, which is essentially the same as sliding a 
two-layer network with the same receptive field as in a 3x3 
convolution (33% cheaper as Szegedy C., et. al. (2016) mention in 
their paper). The third modification was to use auxiliary 
classifiers. Although they did not significantly improve anything 
in the early stages of the training, they resulted in higher 
accuracy towards the end, acting as regularizers. Lastly, grid 
sizes were reduced while expanding the filter banks. For example, 
a dxd grid with k filters, after the reduction would become a 
d/2xd/2 grid with 2k filters.

[InceptionV3 architecture](https://github.com/StMarou/Musculoskeletal-X-Ray-Classification/blob/master/static/inceptionv3.png)

For our problem, in order to improve the predictions of the three 
poorest performing body parts, the features produced by both 
architectures (InceptionV3 and DenseNet169) were utilized. Given 
that these two have different intuitions behind their architectures 
and that they take images of different shape as inputs 
(224x224x3 for the Densenet169 and 299x299x3 for the InceptionV3), 
a larger amount of unique features was produced, which led to 
predictions of higher quality. Also, instead of aggregating the 
estimated probabilities of the two architectures, both pre-trained 
models were combined into a single network, which weighted their 
concatenated feature maps. In this way, the most powerful features 
could ‘cooperate’ to improve the results.

However, given that a little more than half of the training 
instances were used for training and the complexity of the 
resulting model was almost three times bigger DenseNet169’s, the 
model was prone to overfitting. To combat with this problem, the 
dimension of the final layer before the output layer was reduced, 
and dropout was applied. In particular, a 1x1 convolutional layer 
with 512 filters was added on top of each model, followed by a 
dropout layer with a rate of 0.3. Then, a global average pooling 
layer created 512 features from each model, which were then 
concatenated and fed to the output layer. 

[Ensemble architecture](https://github.com/StMarou/Musculoskeletal-X-Ray-Classification/blob/master/static/ensemble.png)

The cohen kappa scores increased for all three body parts, 
with the largest increase observed in shoulder X-Rays (11%). 
Furthermore, the scores of the rest of the metrics were also 
improved (with some exceptions), which means that the ensemble 
architecture could make better predictions with only a little more 
than half of the data.

|        Metric        | DenseNet169 (Hands) | Ensemble (Hands) |
|:--------------------:|:-----:|:-----:|
| Cohen Kappa          | 0.427 | 0.434 |
| Accuracy             | 0.74  | 0.75  |
| Recall (normal)      | 0.93  | 0.97  |
| Recall (abnormal)    | 0.47  | 0.43  |
| Precision (normal)   | 0.71  | 0.71  |
| Precision (abnormal) | 0.83  | 0.90  |
| F1 (normal)          | 0.81  | 0.82  |
| F1 (abnormal)        | 0.60  | 0.59  |

|        Metric        | DenseNet169 (Shoulders) | Ensemble (Shoulders) |
|:--------------------:|:-----:|:-----:|
| Cohen Kappa          | 0.474 | 0.527 |
| Accuracy             | 0.74  | 0.76  |
| Recall (normal)      | 0.74  | 0.77  |
| Recall (abnormal)    | 0.73  | 0.76  |
| Precision (normal)   | 0.74  | 0.77  |
| Precision (abnormal) | 0.73  | 0.76  |
| F1 (normal)          | 0.74  | 0.77  |
| F1 (abnormal)        | 0.73  | 0.76  |

|        Metric        | DenseNet169 (Fingers) | Ensemble (Fingers) |
|:--------------------:|:-----:|:-----:|
| Cohen Kappa          | 0.475 | 0.482 |
| Accuracy             | 0.73  | 0.74  |
| Recall (normal)      | 0.94  | 0.86  |
| Recall (abnormal)    | 0.55  | 0.63  |
| Precision (normal)   | 0.64  | 0.67  |
| Precision (abnormal) | 0.91  | 0.84  |
| F1 (normal)          | 0.76  | 0.75  |
| F1 (abnormal)        | 0.69  | 0.72  |

For the final scores, the ensemble model was used for hand, 
shoulder, and finger X-Rays, and the DenseNet169 model was used 
for the rest. It is observed that all scores increased, except the 
recall score of the normal class and the precision of the 
abnormal class, which remained unchanged.

|        Metric        | Combined Predictions Scores |
|:--------------------:|:-----:|
| Cohen Kappa          | 0.567 | 
| Accuracy             | 0.79  | 
| Recall (normal)      | 0.91  | 
| Recall (abnormal)    | 0.65  | 
| Precision (normal)   | 0.74  | 
| Precision (abnormal) | 0.87  | 
| F1 (normal)          | 0.82  | 
| F1 (abnormal)        | 0.74  | 

## Improvements
Having access to more resources (such as unlimited run-time on 
GPU), the ensemble model could have been fine-tuned using the whole 
training set. Given that it could produce better results with only 
a little more than half of the training instances, using every 
available X-Ray would vastly improve the metrics. Also, the CNN 
model, as well as the number of layers whose weights would been 
set unfrozen could have been tuned. Lastly, more ensembles with 
different pre-trained architectures could have been created, whose 
predicted probabilities could be aggregated (such as taking their 
average) to further increase the scores and the generalization 
capabilities of the models.

## References
[1] Huang, G., Liu, Z., Van Der Maaten, L. and Weinberger, K.Q., 2017. 
Densely connected convolutional networks. In *Proceedings of the 
IEEE conference on computer vision and pattern recognition* 
(pp. 4700-4708).

[2] Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. 
Rethinking the inception architecture for computer vision. 
In *Proceedings of the IEEE conference on computer vision and 
pattern recognition 2016* (pp. 2818-2826).

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., 
Anguelov, D., Erhan, D., Vanhoucke, V. and Rabinovich, A., 2015. 
Going deeper with convolutions. In *Proceedings of the IEEE 
conference on computer vision and pattern recognition* (pp. 1-9).


