
import torch

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s


def place_tensors_on_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """
    Place n tensors from a tensor of shape (nB, B) onto the diagonal of a larger tensor
    of shape (nB, nB), filling other positions with zeros.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (nB, B).
        
    Returns:
        torch.Tensor: Output tensor of shape (nB, nB) with the n tensors on the diagonal.
    """
    nB, B = tensor.shape
    n = nB // B  # Number of smaller tensors
    
    # Create an output tensor filled with zeros
    result = torch.zeros((nB, nB), dtype=tensor.dtype, device=tensor.device)
    
    # Fill the diagonal blocks
    for i in range(n):
        result[i * B:(i + 1) * B, i * B:(i + 1) * B] = tensor[i * B:(i + 1) * B]
    
    return result