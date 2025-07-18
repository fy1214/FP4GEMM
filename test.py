import torch
import torch.nn as nn

class AdvancedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(AdvancedModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        ).dtype(torch.bfoat16)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
         ).dtype(torch.float)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
         ).dtype(torch.bfoat16)
        self.fc = nn.Linear(hidden_size//2, output_size).dtype(torch.uint8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out

def _allocate_buffers_for_parameters(
    input_params
):
    param_and_grad_dtype_to_params = {}
    param_and_grad_dtype_to_offsets = {}
    param_and_grad_dtype_to_indices = {}

    # Group parameters by their gradient type.
    for param in input_params:
        assert param.requires_grad

        param_dtype = param.dtype
        if is_float8tensor(param):
            # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
            # dtype (usually a higher precision dtype such as bfloat16), but its actual
            # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
            # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
            # it is necessary to use torch.uint8, not the "fake" dtype got from
            # "param.dtype".
            param_dtype = torch.uint8
        grad_dtype = torch.float

        params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
        params.append(param)
        param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

        # Get the index of each param among the params with same dtype, if a param is fp8,
        # use its "fake" high precision dtype to find which params have same dtype with it.
        # For example:
        #     Case 1:
        #         params = [p1(bf16), p2(bf16), p3(bf16), p4(bf16)]
        #         param_and_grad_dtype_to_indices = {
        #             (torch.bfloat16, torch.float32): [0, 1, 2, 3],
        #         }
        #     Case 2:
        #         params = [p1(bf16), p2(fp8), p3(fp8), p4(bf16)]
        #         param_and_grad_dtype_to_indices = {
        #             (torch.bfloat16, torch.float32): [0, 3],
        #             (torch.uint8, torch.float32): [1, 2],
        #         }
        # We need these indices to load a non-native-fp8 checkpoint in native-fp8 mode.
        offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
        param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
        indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
        indices.append(offset)
        param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices
    
    print(param_and_grad_dtype_to_indices)


if __name__ == '__main__':
    model = AdvancedModel(1024, 4096, 512)
    param = []
    for name, param in model.named_parameters():
        param.append(param)
    _allocate_buffers_for_parameters(param)

