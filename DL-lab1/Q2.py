# Write down the observations from the plot for all the above functions in the code.

# Sigmoid: S-shaped, output 0–1.
# Tanh: S-shaped, output -1 to 1, zero-centered.
# ReLU: 0 for negative values, linear for positive values.
# Leaky ReLU: Small output for negative values, linear for positive values.
# Softmax: Converts outputs into probabilities.


# What are the min and max values for the functions?

# 	1.sigmoid: 0-1
# 	2.Tanh: -1 to +1
# 	3.ReLU: 0 to infinity
# 	4. Leaky ReLU: negative infinity to positive infintity
# 	5. Softmax: 0 to 1(sum of output 1)

# Is the output of the function zero-centred?

# 	1.sigmoid: no
# 	2.Tanh: yes
# 	3.ReLU: no
# 	4. Leaky ReLU: no
# 	5. Softmax: no

# What happens to the gradient when the input values are too small or too big?

# 	1.sigmoid:
# 			1.Gradient decent almost zero for posive and negative values.Causes vanishing gradients problems
# 	2.Tanh:
# 			1.Gradient is highest near zero. for every higher value of positive and negatve input gradient is almost zero
# 	3.ReLU:
# 			1.for positive input gradient 1, For negative input gradient is zero.
# 	4. Leaky ReLU:
# 			1. forpostive input gradient is zero, for negative input gradient is small(ex:0.01), prevent dead neurons.
# 	5. Softmax:
# 			1.Gradient depends on all output probabilities. Used with Cross-Entropy Loss for stable training.

# What is the relationship between sigmoid and tanh?

#1. tanh output values beween -1 to 1 byt sigmoid output va;ues 0 to 1,
#2. both have vanishing gradient problem,t
#3. tanh is zero centred usually trains NN faster than sigmoid func,
#4. Both S shaped



