# Issue #003: Energy Conservation Violations in Symplectic Integrator

**Version:** v1.0.0  
**Priority:** Critical  
**Type:** Numerical Stability  
**Component:** SymplecticKoopmanOperator (removed in current version)  
**Created:** 2024-01-10  

## Description
Despite adaptive timestep control, energy conservation violations persist at ~1e-3 to 1e-4 level, far above the 1e-6 tolerance. The physics simulation is not truly symplectic, leading to drift and instability.

## Impact
- Physics simulation lacks fundamental energy conservation
- Long-term instability in temporal dynamics
- Violates core principle of Hamiltonian mechanics
- Affects system stability and predictability

## Root Cause Analysis
1. Possible mismatch between assumed Hamiltonian (H = 0.5*(p² + q²)) and actual learned dynamics
2. Normalization operations may break symplectic structure
3. Adaptive timestep may be insufficient for numerical precision required
4. Integration scheme may not be truly symplectic

## Reproduction
```python
from active_inference_agent import SymplecticKoopmanOperator
import torch

# Test energy conservation
operator = SymplecticKoopmanOperator(latent_dim=32)
test_passed, results = operator.isolated_test_energy_conservation(num_steps=100)
print(f"Energy drift: {results['max_drift']}")  # Often > 1e-6
```

## Acceptance Criteria
- [ ] Energy conservation within 1e-6 tolerance
- [ ] Stable evolution over 1000+ integration steps
- [ ] No unbounded growth in magnitudes
- [ ] Pass isolated energy conservation tests
- [ ] Maintain symplectic structure mathematically

## Suggested Solutions
1. **Implement higher-order symplectic integrators** (e.g., 4th order Yoshida)
2. **Learn the actual Hamiltonian** instead of assuming harmonic oscillator
3. **Remove normalization from integration step** that breaks symplectic structure
4. **Use exact energy conservation constraints** during training
5. **Implement implicit integrators** for better stability

## Test Reference
- Custom energy conservation tests in SymplecticKoopmanOperator
- `isolated_test_energy_conservation()` method

## Related Issues
- #008 (Aggressive learning rates affecting stability)

## Status
- Currently: SymplecticKoopmanOperator removed from main version
- Needs: Complete rework or replacement with stable alternative

## Labels
- critical
- physics
- numerical-stability
- symplectic-integrator