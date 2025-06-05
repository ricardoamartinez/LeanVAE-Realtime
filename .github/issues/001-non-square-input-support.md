# Issue #001: Non-Square Input Support Missing

**Version:** v1.0.0  
**Priority:** High  
**Type:** Feature Gap  
**Component:** PhysicsInformedWaveletTransform  
**Created:** 2024-01-10  

## Description
The wavelet transform implementation assumes square inputs. Non-square inputs (e.g., 640x480) will fail or produce incorrect results.

## Impact
- Limits real-world applications with standard video resolutions
- Cannot process standard camera feeds without preprocessing
- Reduces flexibility for different input formats

## Reproduction
```python
from active_inference_agent import PhysicsInformedWaveletTransform
import torch

# This will fail or produce incorrect results
transform = PhysicsInformedWaveletTransform(input_size=64)
non_square_input = torch.randn(1, 3, 64, 48)  # Non-square
output = transform.forward_2d_spatial_wavelet(non_square_input)  # Error or wrong output
```

## Acceptance Criteria
- [ ] Support rectangular inputs of any aspect ratio
- [ ] Maintain energy conservation for non-square inputs
- [ ] Add proper padding or adaptive processing
- [ ] Update all related tests
- [ ] Document supported input formats

## Suggested Implementation
1. Implement proper handling for rectangular inputs
2. Add padding strategies (zero, reflection, etc.)
3. Modify inverse transform to handle non-square coefficients
4. Update spatial feature calculation

## Test Reference
`test_wavelet_transform.py::TestPhysicsInformedWaveletTransform::test_edge_cases`

## Related Issues
- #002 (Fixed 3-channel requirement)

## Labels
- bug
- enhancement
- high-priority
- wavelet-transform