from typing import Dict, Any
import torch

from asparagus import utils

# --------------- ** Checking system property convention ** ---------------

def check_fragmented_properties(
    batch: Dict[str, Any],
    ml_fragment: int = 0,
    mm_fragment: int = 1,
) -> Dict[str, Any]:
    """
    Check system properties in batch for the correct property convention and,
    eventually, manipulate properties accordingly

    Parameters:
    -----------
    batch: dict[str, any]
        Data batch with structural properties
    ml_fragment: int, optional, default 0
        Atomic fragment number for ML atoms
    mm_fragment: int, optional, default 1
        Atomic fragment number for MM atoms

    Return:
    -------
    dict[str, any]
        Updated data batch

    """

    # Only for fragmented systems
    if batch['fragmented']:

        # Check if system properties are already converted
        Nsys = batch['atoms_number'].shape[0]
        ml_selection = batch['fragment_numbers'] == ml_fragment
        if torch.all(ml_selection):
            batch['mlmm_atoms_number'] = batch['atoms_number']
            batch['mlmm_atomic_numbers'] = batch['atomic_numbers']
            batch['mlmm_sys_i'] = batch['sys_i']
            batch['mlmm_positions'] = batch['positions']
            batch['ml_sys_p'] = torch.arange(
                batch['fragment_numbers'].shape[0],
                device=batch['fragment_numbers'].device,
                dtype=batch['fragment_numbers'].dtype
            ).detach()
            batch['ml_idx_p'] = batch['ml_sys_p'].clone()
            return batch

        # Get ML atom numbers
        ml_atoms_number = torch.zeros(
            Nsys,
            device=batch['atoms_number'].device,
            dtype=batch['atoms_number'].dtype,
        ).scatter_add(
            0,
            batch['sys_i'],
            ml_selection.to(dtype=torch.int64)
        )

        # If necessary, get ML atoms pointer lists and add to batch
        ml_sys_p = torch.arange(
            batch['fragment_numbers'].shape[0],
            device=batch['fragment_numbers'].device,
            dtype=batch['fragment_numbers'].dtype
        )[ml_selection]
        ml_idx_p = torch.full_like(
            batch['fragment_numbers'],
            -1,
            device=batch['fragment_numbers'].device,
            dtype=batch['fragment_numbers'].dtype)
        for ia, ai in enumerate(ml_sys_p):
            ml_idx_p[ai] = ia
        batch['ml_sys_p'] = ml_sys_p.detach()
        batch['ml_idx_p'] = ml_idx_p.detach()
        
        # Re-assign basic system properties
        batch['mlmm_atoms_number'] = batch['atoms_number'].clone()
        batch['atoms_number'] = ml_atoms_number
        batch['mlmm_atomic_numbers'] = batch['atomic_numbers'].clone()
        batch['atomic_numbers'] = batch['atomic_numbers'][ml_sys_p]
        batch['mlmm_sys_i'] = batch['sys_i'].clone()
        batch['sys_i'] = batch['sys_i'][ml_sys_p]
        batch['mlmm_positions'] = batch['positions'].clone()
        # ML positions needs to be re-assigned when forces via back-propagation
        # are requested and 'requires_grad' is set True. E.g. in 
        # model.forward():
        # if batch['fragmented']:
        #     batch['mlmm_positions'].requires_grad_(True)
        #     batch['positions'] = batch['mlmm_positions'][batch['ml_sys_p']]
        # else:
        #     batch['positions'].requires_grad_(True)
        batch['positions'] = batch['mlmm_positions'][ml_sys_p]

    else:
        
        batch['mlmm_atoms_number'] = batch['atoms_number']
        batch['mlmm_atomic_numbers'] = batch['atomic_numbers']
        batch['mlmm_sys_i'] = batch['sys_i']
        batch['mlmm_positions'] = batch['positions']

    return batch
    
    
