import torch

def to_onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot

y = torch.tensor([0, 1, 2, 2])

y_enc = to_onehot(y, 3)

print('one-hot encoding:\n', y_enc)

# Expected output:
#one-hot encoding:
# tensor([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.],
#        [0., 0., 1.]])


Z = torch.tensor( [[-0.3,  -0.5, -0.5],
                   [-0.4,  -0.1, -0.5],
                   [-0.3,  -0.94, -0.5],
                   [-0.99, -0.88, -0.5]])

print(Z)

# Expected output:
#tensor([[-0.3000, -0.5000, -0.5000],
#        [-0.4000, -0.1000, -0.5000],
#        [-0.3000, -0.9400, -0.5000],
#        [-0.9900, -0.8800, -0.5000]])
        
def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

smax = softmax(Z)
print('softmax:\n', smax)

# Expected output:
#softmax:
#tensor([[0.3792, 0.3104, 0.3104],
#        [0.3072, 0.4147, 0.2780],
#        [0.4263, 0.2248, 0.3490],
#        [0.2668, 0.2978, 0.4354]])

def to_classlabel(z):
    return torch.argmax(z, dim=1)

print('predicted class labels: ', to_classlabel(smax))
print('true class labels: ', to_classlabel(y_enc))

# Expected output:
# predicted class labels:  tensor([0, 1, 0, 2])
# true class labels:  tensor([0, 1, 2, 2])

# In PyTorch:

import torch.nn.functional as F

#Note that nll_loss takes log(softmax) as input:

print(F.nll_loss(torch.log(smax), y, reduction='none'))

# Expected output:
# tensor([0.9698, 0.8801, 1.0527, 0.8314])

# Note that cross_entropy takes logits as input:

print(F.cross_entropy(Z, y, reduction='none'))

# Expected output:
# tensor([0.9698, 0.8801, 1.0527, 0.8314])

# Defaults

# By default, nll_loss & cross_entropy are already returning the average over 
# training examples, which is useful for stability during optimization.

print(F.cross_entropy(Z, y))

# Expected output
# tensor(0.9335)

#print(torch.mean(cross_entropy(smax, y_enc)))
print(torch.mean(F.cross_entropy(smax, y_enc)))

# Expected output
# tensor(0.9335) # I'm getting something else....?





