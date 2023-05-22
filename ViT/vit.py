import torch
import torch.nn as nn


class Patches(nn.Module):
    def __init__(self, size):
        super(Patches).__init__()
        self.size = size

    def forward(self, img):
        patches = img.unfold(1, self.size, self.size).unfold(2, self.size, self.size).unfold(3, self.size, self.size)
        return patches




class PatchEncoder(nn.Module):
    def __init__(self, patchnum, proj_dim):
        super(PatchEncoder).__init__()

        self.patchnum = patchnum
        self.proj = nn.Linear(patchdim, proj_dim)
        self.pos_embed = nn.Embedding(patchnum, proj_dim)

    def forward(self, patch):
        positions = torch.range(0, self.patchnum, step=1)
        encoded = self.proj(patch) + self.pos_embed(positions)
        return encoded

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer).__init__()



class EfficientViTTransformer(nn.Module):


    def __init__(self, patch_size, projdim):
        super().__init__()
        # Architecture idea, based off of CrossEfficient ViT
        # EfficientNet feature extractor on patches
        # Linear Projection into Transformer Encoder
        # Output into Cross Attention

        self.patch_size = patch_size
        num_patches = (7 // patch_size) ** 2
        classes = 2

        self.patches = Patches(patch_size)
        self.encoded = PatchEncoder(num_patches, projdim)
        self.efficient = EfficientNet() #create copy that handles small size



    def forward(self, x):
        x = self.efficient(x)
        x = self.patches(x)
        x = self.encoded(x)
        


