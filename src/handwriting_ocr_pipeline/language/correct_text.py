# from neuspell import BertChecker
# import torch

# # Monkey-patch torch.load to set weights_only=False
# original_torch_load = torch.load
# def patched_load(*args, **kwargs):
#     if "weights_only" not in kwargs:
#         kwargs["weights_only"] = False
#     return original_torch_load(*args, **kwargs)

# torch.load = patched_load

# checker = BertChecker()
# checker.from_pretrained()

# device = "cpu"
# model.to(device)

# def correct(text):
#     # Encode input
#     t = "Deadback promotion ensures at least one of the four deadlock condition's Grutual exclusion, hold-and-wait, no exemption, circular wait."
#     s = checker.correct(t)
#     print(s)
#     return s