import random
import math
import torch
random.seed(77)
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
    def __init__(self, input_size, min_number_of_patches = 16, min_aspect_ratio=0.3, max_aspect_ratio=3.33, max_masking_factor = 0.4):
        #super().__init__()
        if not isinstance(input_size, tuple):
           input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.n_patches = self.height * self.width

        self.min_number_of_patches = min_number_of_patches


        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.max_mask_patches = min(self.n_patches * max_masking_factor, 75)

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
            aspect_ratio = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if height <= self.height and width <= self.width:
                #choose a random top left corner
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                #mask the block
                for i in range(height):
                    for j in range(width):
                        if mask[top + i, left + j] == False:
                            mask[top + i, left + j] = True
                            n_new_patches += 1
                
                if n_new_patches > 0:
                    break

        return mask, n_new_patches
    
    def generate_mask(self, batch_size = 32):
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
        mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        #print(mask.shape, mask)
        n_masked_patches = 0
        while n_masked_patches < self.max_mask_patches:
            mask, n_new_patches = self._mask(mask, self.max_mask_patches - n_masked_patches)
            n_masked_patches += n_new_patches
        return mask

