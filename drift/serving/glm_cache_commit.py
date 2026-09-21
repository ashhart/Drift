"""Validate cache destinations and verify packed bytes before an applied receipt."""


def validate_destinations(part, page, slot, rows):
    import torch
    if page.shape != (rows,) or slot.shape != (rows,):
        raise ValueError('receiver destination index shape mismatch')
    if page.dtype != torch.long or slot.dtype != torch.long:
        raise ValueError('receiver destination index dtype mismatch')
    if page.device != part.device or slot.device != part.device:
        raise ValueError('receiver destination index device mismatch')
    if ((page < 0) | (page >= part.shape[0])).any().item():
        raise ValueError('receiver page index outside cache')
    if ((slot < 0) | (slot >= part.shape[1])).any().item():
        raise ValueError('receiver slot index outside cache')
    if torch.unique(page * part.shape[1] + slot).numel() != rows:
        raise ValueError('receiver destination slots alias')


def commit_cache(staged, packed_width):
    import torch
    on_gpu = any(part.is_cuda for part, _, _, _ in staged)
    if on_gpu:
        torch.cuda.synchronize()
    for part, page, slot, packed in staged:
        part[page, slot, :packed_width] = packed
    checks = [(part[page, slot, :packed_width] == packed).all()
              for part, page, slot, packed in staged]
    if on_gpu:
        torch.cuda.synchronize()
    if not all(check.item() for check in checks):
        raise RuntimeError('receiver cache readback mismatch')
