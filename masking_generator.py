import random
import math
import torch
import numpy as np

class MaskingGenerator:
    """Generate mask for MIM task.
    
    Parameters
    ----------
    input_size : int
        Size of the image.
    min_number_of_patches : int
        Minimum number of patches to be masked.
    min_aspect_ratio, max_aspect_ratio : float
        Minimum and maximum aspect ratio of the block.
    max_masking_factor : float
        Maximum percentage of the image that can be masked.

    Attributes
    ----------

    """
    def __init__(self, input_size, n_masking_patches = 75, min_number_of_patches = 4, min_aspect_ratio=0.3, max_aspect_ratio=3.33, max_masking_factor = 0.4):
        #super().__init__()
        if not isinstance(input_size, tuple):
           input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.n_patches = self.height * self.width
        
        self.n_masking_patches = n_masking_patches
        self.min_number_of_patches = min_number_of_patches
        self.max_number_of_patches = n_masking_patches


        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.log_aspect_ratio = (math.log(min_aspect_ratio), math.log(max_aspect_ratio))

    def _mask(self, mask, max_mask_patches, max_attempts = 10):
        """Create a mask made of a single block.
        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor.
        max_mask_patches : int 
            Maximum number of patches to be masked.
        max_attempts : int
            Number of attempts to create a mask before returning.

        Returns
        -------
        torch.Tensor
            Mask tensor.
        n_new_patches : int
            Number of patches that were masked.
        """

        n_new_patches = 0
        for attempt in range(max_attempts):
            #choose a random block size
            target_area = random.uniform(self.min_number_of_patches, max_mask_patches)
            #aspect_ratio = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if height <= self.height and width <= self.width:
                #choose a random top left corner
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                n_masked = mask[top: top + height, left: left + width].sum()
                if 0 < height * width - n_masked <= max_mask_patches:
                    #mask the block
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if mask[i,j] == 0:
                                mask[i,j] = 1
                                n_new_patches += 1
                
                if n_new_patches > 0:
                    break

        return n_new_patches
    
    def __call__(self):
        """Generate mask for MIM task.
        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        torch.Tensor
            Mask tensor.
        """
        mask = np.zeros(shape = (self.height, self.width), dtype = int)
        n_masked_patches = 0
        while n_masked_patches < self.n_masking_patches:
            max_mask_patches = self.n_masking_patches - n_masked_patches
            max_mask_patches = min(max_mask_patches, self.max_number_of_patches)

            n_new_patches = self._mask(mask, max_mask_patches)
            if n_new_patches == 0:
                break
            else:
                n_masked_patches += n_new_patches
        
        return mask

