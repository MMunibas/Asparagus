from typing import Optional, List, Any

import numpy as np

import torch

from asparagus import utils


def encode_str2int(
    name: str,
    nchar: Optional[int] = 4,
) -> int:
    """
    Encode string of length 4 to a max 12-digit number of ncharx3 blocks of the
    respective ASCII list number of each character.
    
    Parameters
    ----------
    name: str
        Input name tag
    nchar: int
        Number of character to encode

    Returns
    -------
    int
        Encoded name tag

    """

    return int(''.join([f'{ord(char):03d}' for char in name[:nchar]]))


def decode_int2str(
    code: int,
    nchar: Optional[int] = 4,
) -> str:
    """
    Decode max 12-digit number into a 4 character string. Each of the ncharx3 
    blocks of numbers are the ASCII list numbers of the respective character.
    
    Parameters
    ----------
    code: int
        Encoded name tag number
    nchar: int
        Number of number blocks to decode

    Returns
    -------
    str
        Decoded Name tag

    """
    numbers = []
    str_code = str(code)
    for ii in range(nchar):
        if ii:
            numbers.append(int(str_code[-(ii + 1)*3:-ii*3]))
        else:
            numbers.append(int(str_code[-(ii + 1)*3:]))
    return ''.join(chr(numb) for numb in numbers[::-1])


def detach_tensor(
    x: torch.Tensor,
) -> np.ndarray:
    """
    Detach a torch tensor from the computational graph

    Parameters
    ----------
    x: Any
        Input variable to detach

    Returns
    -------
    Any
        Detached input variable

    """
    if utils.in_cuda(x):
        x.cpu()
        x.detach().numpy()
    else:
        x.detach().numpy()
    return x


def flatten_array_like(
    x: List[Any],
) -> List[Any]:
    # In case x is a "list" of characters aka a string
    if utils.is_string(x):
        yield x
    else:
        for xi in x:
            if utils.is_array_like(xi):
                for xj in flatten_array_like(xi):
                    yield xj
            else:
                yield xi


def segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: Optional[int] = None,
    device: Optional[str] = 'cpu',
    debug: Optional[bool] = False,
) -> torch.Tensor:
    """
    Adapted from :
        https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be
    Analogous to tf.segment_sum :
        https://www.tensorflow.org/api_docs/python/tf/math/segment_sum

    Parameters
    ----------
    data: torch.Tensor
        A pytorch tensor of the data for segmented summation.
    segment_ids: torch.Tensor, shape(N)
        A 1-D tensor containing the indices for the segmentation.
    num_segments: int, optional, default None
        The number of segments. If None and with the requirement of a sorted
        'segment_ids', this number should be the last element plus 1.

    Returns
    -------
    torch.Tensor
        A tensor of the same type as data containing the results of the
        segmented summation.

    """

    if debug:

        if not all(
            segment_i <= segment_j for segment_i, segment_j
            in zip(segment_ids[:-1], segment_ids[1:])
        ):

            raise AssertionError("Elements of 'segment_ids' must be sorted")

        if len(segment_ids.shape) != 1:
            raise AssertionError("'segment_ids' have to be a 1-D tensor")

        if data.shape[0] != segment_ids.shape[0]:
            raise AssertionError(
                "'data' and 'segment_ids'should be the same size at "
                + f"dimension 0 but are ({data.shape[0]:d}) and "
                + f"({segment_ids.shape[0]}).")

    if num_segments is None:
        num_segments = segment_ids[-1] + 1  # len(torch.unique(segment_ids))
    return unsorted_segment_sum(
        data, segment_ids, num_segments, device=data.device)


def unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    device: Optional[str] = 'cpu',
    debug: Optional[bool] = False,
) -> torch.Tensor:
    """
    Computes the sum along segments of a tensor. Analogous to
    tf.unsorted_segment_sum.

    Parameters
    ----------
    data: torch.Tensor
        A tensor whose segments are to be summed.
    segment_ids: torch.Tensor, shape(N)
        The segment indices tensor.
    num_segments: int
        The number of segments.

    Returns
    -------
    torch.Tensor
        A tensor of same data type as the data argument.

    """

    if debug:

        msg = "'segment_ids.shape' should be a prefix of 'data.shape'!"
        assert all([i in data.shape for i in segment_ids.shape]), msg

        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(
                segment_ids.shape[0], *data.shape[1:]).to(device)

        msg = "'data.shape' and 'segment_ids.shape' should be equal!"
        assert data.shape == segment_ids.shape, msg

    else:

        s = torch.prod(
            torch.tensor(data.shape[1:], dtype=torch.int64, device=device)
            ).to(device)
        segment_ids = segment_ids.repeat_interleave(s).view(
            segment_ids.shape[0], *data.shape[1:]).to(device)

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(
        *shape, dtype=data.dtype, device=device).scatter_add(
            0, segment_ids, data)

    return tensor


def gather_nd(
    params: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    The input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    This function comes from:
    https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502/6

    Parameters
    ----------
    params: torch.Tensor
        A tensor of any shape.
    indices: torch.Tensor
        A 2d tensor in the form of [[a,b,..,c],...]

    """

    # Generate indices
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for irev in range(ndim):
        i = ndim - irev - 1
        idx = idx + indices[i] * m
        m *= params.size(i)

    params = params.reshape((-1, *tuple(torch.tensor(params.size()[ndim:]))))
    return params[idx]


def _broadcast(
    index: torch.Tensor,
    data: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Computes the sum along the segmented tensor according to index array.
    Code gladly taken and modified from the source:
        https://github.com/rusty1s/pytorch_scatter

    Parameters
    ----------
    index: torch.tensor
        Index array to assign source data to output tensor
    data: torch.tensor
        Source segmented data tensor
    dim: int, optional, default 0
         The axis along which to index

    Returns
    -------
    torch.tensor
        Adjusted index array of matching shape with respect to source

    """
    if dim < 0:
        dim = data.dim() + dim
    if index.dim() == 1:
        for _ in range(0, dim):
            index = index.unsqueeze(0)
    for _ in range(index.dim(), data.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(data.size())
    return index


def scatter_sum(
    data: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes the sum along the segmented tensor according to index array.
    Code gladly taken and modified from the source:
        https://github.com/rusty1s/pytorch_scatter

    Parameters
    ----------
    data: torch.Tensor
        Source segmented data tensor
    index: torch.Tensor
        Index array to assign source data to output tensor
    dim: int, optional, default -1
        Dimension of output tensor to add source data
    out: torch.Tensor, optional, default None
        Output tensor to which source data are added.
        If None, output tensor shape is predicted (takes more time) and
        initialized as zero tensor with dtype and device of source data.
    dim_size: int, optional, default None
        Size of output tensor shape at 'dim'.
        If None, it will be predicted by maximum number in index array (takes
        more time).
        Ignored if a parameter for 'out' is provided.

    Returns
    -------
    torch.Tensor
        Output tensor to which source data are added

    """
    index = _broadcast(index, data, dim)
    if out is None:
        size = list(data.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=data.dtype, device=data.device)
        return out.scatter_add_(dim, index, data)
    else:
        return out.scatter_add_(dim, index, data)
