import torch
from torch.nn import functional as F

def masked_softmax(x, mask_length):
    def _seq_mask(x, mask_len, val=0):
        mxLen = x.size(1)
        mask = torch.arange((mask_length), 
                            dtype=torch.float32,
                            device=x.device)[None, :] < mask_length[:, None]
        x[~mask] = val
        return x

    if mask_length is None:
        return F.softmax(x, dim=-1)
    else:
        shape = x.shape
        if mask_length.dim() == 1:
            mask_length = torch.repeat_interleave(mask_length, shape[1])
        else:
            mask_length = mask_length.reshape(-1)
    x  =_seq_mask(x.reshape(-1, shape[-1]), mask_length, val=1e-6)
    return F.softmax(x.reshape(shape), dim=-1)

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
