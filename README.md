
# Deep learning Playbook

>"Successfully applying deep learning techniques requires more than just a good knowledge of what algorithms exist and the principles that explain how they work...

> ...A good machine learning practitioner also needs to know how to choose an algorithm for a particular application and how to monitor and respond to feedback obtained from experiments in order to improve a machine learning system. During day to day development of machine learning systems, practitioners need to decide whether to gather more data, increase or decrease model capacity, add or remove regularizing features, improve the optimization of a model, improve approximate inference in a model, or debug the software implementation of the model. All of these operations are at the very least time-consuming to try out, so it is important to be able to determine the right course of action rather than blindly guessing." - deeplearningbook.org

---

# Table of Contents 

* Choosing appropriate activation functions
* Weight initialisation
* Hyperparameter tuning
    * Learning rate
* Strategies to improve performance
	* Ensembles 
* Generalisation and avoiding overfitting
* Data preprocessing & preparation
* Cost functions
* Optimisation
* Monitoring the training
* Saving and Loading models
* On GPUs
* Trouble-shooting & debugging strategies
* Applied Computer vision
	* Data augmentation
	* Transfer learning 
* Applied Natural language processing
* Advice from top practitioners

---


# Choosing appropriate activation functions

Every activation function (or non-linearity) takes a single number and performs a certain fixed mathematical operation on it.

There are several activation functions you may encounter in practice:
* Sigmoid
* Tanh
* ReLU
* ReLU variants - Leaky ReLU
* Maxout


__Tanh and sigmoid__

Sigmoid and Tanh are the two most commonly used activation functions. However, they are not necessarily good choices as activation functions for the hidden units.

The reasons are:
* They saturate quickly when the output value is not a narrow region (around 0), i.e. the derivative almost vanishes. This means that the network easily gets stuck or converges slowly when the pre-activated values fall outside of the region.

* There's no reason that the hidden units should be restricted to (0,1) as in the Sigmoid function or (-1,1) as in the Tanh function.

* Choose a good nonlinear function: in practice tanh tends to work better than sigmoid. The rectified linear function is also a good choice

* Derivative of the sigmoid function is always smaller than 1


__Sigmoid__

It takes a real-valued number and “squashes” it into range between 0 and 1. In particular, large negative numbers become 0 and large positive numbers become 1. In practice, the sigmoid non-linearity has recently fallen out of favor and it is rarely ever used. It has two major drawbacks:
* Sigmoids saturate and kill gradients.
* Sigmoid outputs are not zero-centered.

__Tanh__

It squashes a real-valued number to the range [-1, 1].
Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity. Some evidence suggests it outperforms sigmoid neurons

tanh is just a rescaled and shifted sigmoid, but better for many models
    • Initialization: values close to 0
    • Convergence: faster in practice
    • Nice derivative (similar to sigmoid)    

shape approximates the sigmoid function, but ranges from -1 to 1 instead of zero to one, thereby facilitating both positive and negative activations

ReLU
rectified linear unit or rectified linear neuron

* Computationally simpler relative to sigmoid or tanh, but in a network can approximate their performance and nevertheless compute any function
* The Rectified Linear Unit has become very popular in the last few years.
* It computes the function f(x)=max(0,x)

* In other words, the activation is simply thresholded at zero (see image above on the left). There are several pros and cons to using the ReLUs:
        (+) It was found to greatly accelerate (e.g. a factor of 6 in Krizhevsky et al.) the convergence of stochastic gradient descent compared to the sigmoid/tanh functions. It is argued that this is due to its linear, non-saturating form.
        (+) Compared to tanh/sigmoid neurons that involve expensive operations (exponentials, etc.), the ReLU can be implemented by simply thresholding a matrix of activations at zero.
        (-) Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be “dead” (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue.

Reason
1. Fast to compute
2. Biological reason
3. Infinite sigmoid with different biases
4. Solution for vanishing gradient

ReLU variants:
LReLU & PReLU
Maxout

__Leaky ReLU__
Leaky ReLUs are one attempt to fix the “dying ReLU” problem. Instead of the function being zero when x < 0, a leaky ReLU will instead have a small negative slope (of 0.01, or so).

__PReLU neurons__
The slope in the negative region can also be made into a parameter of each neuron, as seen in PReLU neurons,

__Maxout__
One relatively popular choice is the Maxout neuron (introduced recently by Goodfellow et al.) that generalizes the ReLU and its leaky version. The Maxout neuron computes the function max(wT1x+b1,wT2x+b2)
Notice that both ReLU and Leaky ReLU are a special case of this form (for example, for ReLU we have w1,b1=0. The Maxout neuron therefore enjoys all the benefits of a ReLU unit (linear regime of operation, no saturation) and does not have its drawbacks (dying ReLU).

However, unlike the ReLU neurons it doubles the number of parameters for every single neuron, leading to a high total number of parameters. This concludes our discussion of the most common types of neurons and their activation functions. As a last comment, it is very rare to mix and match different types of neurons in the same network, even though there is no fundamental problem with doing so.



__Takeaway__

Choose the right activation function (linear, sigmoid, tanh, relu etc) and the right loss/divergence function (MSE, Cross entropy, binary cross-entropy etc) based on your data

TLDR: “What neuron type should I use?” Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of “dead” units in a network.
If this concerns you, give Leaky ReLU or Maxout a try.

Never use sigmoid.
Try tanh, but expect it to work worse than ReLU/Maxout.

Avoid Sigmoid's, TanH's gates they are expensive and get saturated and may stop back propagation. In fact the deeper your network the less attractive Sigmoid's and TanH's are.

Use the much cheaper and effective ReLU's and PreLU's instead. As mentioned in Deep Sparse Rectifier Neural Networks they promote sparsity and their back propagation is much more robust. Don't use ReLU or PreLU's gates before max pooling, instead apply it after to save computation. Don't use ReLU's they are so 2012. Yes they are a very useful non-linearity that solved a lot of problems. However try fine-tuning a new model and watch nothing happen because of bad initialization with ReLU's blocking backpropagation. Instead use PreLU's with a very small multiplier usually 0.1. Using PreLU's converges faster and will not get stuck like ReLU's during the initial stages. ELU's are still good but expensive.



---
---













