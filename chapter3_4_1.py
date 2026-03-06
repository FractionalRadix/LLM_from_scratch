import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 # Normally input and output size are the same, here it's less for educational purposes.

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False) 
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# Hm... so the maths is...
# attn_Score_22 = query_2 * keys_2 
#               = query_2 * keys[1] 
#               = query_2 * ((inputs @ W_key)[1])
#               = (x_2 @ W_query) * ((inputs @ W_key)[1])
#               = ((inputs[1]) @ W_query) * ((inputs @ W_key)[1])

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
# context_vec_2 = attn_weights_2 @ (inputs @ W_value)
#               = s(attn_scores_2) @ (inputs @ W_value)
#               = s(query_2 @ keys.T) @ (inputs @ W_value)
#               = s((x_2 @ W_query) @ keys.T) @ (inputs @ W_value)
#               = s((x_2 @ W_query) @ (inputs @ W_key).T) @ (inputs @ W_value)
#                   where x_2 = inputs[1]
# So: you take a word (embedding vector) from the input.
#     You multiply this with the query weights, and then with the (transposed) product of the inputs and the key weights.
#     You normalize this (using softmax, and dividing by the square root of a dimension for efficiency during the training process).
#     You multiply this normalized input-queryweight-times-inputs-keyweight thing with the product of inputs and values.
#
# The keys are supposedly used to match the queries. This is then used to retrieve the associated values.


