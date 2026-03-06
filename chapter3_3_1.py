import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# Calculate the attention weights

query = inputs[1] # Arbitrarily choose the second token.
                  # We are going to calculate the intermediate attention scores.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
  attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Calculate the context vector 

query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

# -- Dot product using for loops (slow).
#attn_scores = torch.empty(6, 6)
#for i, x_i in enumerate(inputs):
#  for j, x_j in enumerate(inputs):
#    attn_scores[i, j] = torch.dot(x_i, x_j)
#print(attn_scores)

# -- Same dot product, using PyTorch's (?) matrix multiplication.
attn_scores = inputs @ inputs.T
print(attn_scores)
# -- Softmax for normalization. 
#  "dim=-1" means normalization should be applied along the last dimension of the tensor.
#  In this case, that is the rows.
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# Verify that the rows now sum to 1:
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])    
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# So...
# - attention scores is "inputs @ inputs.T"
# - attention weights = attention scores normalized over rows, using softmax
# - context vectors = attention weights @ inputs


