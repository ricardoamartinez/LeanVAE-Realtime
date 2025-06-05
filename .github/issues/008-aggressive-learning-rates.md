# Issue #008: Aggressive Learning Rates Causing Training Instability

**Version:** v1.0.0  
**Priority:** High  
**Type:** Training Stability  
**Component:** UltraFast60FpsLeanVAE  
**Created:** 2024-01-10  

## Description
Learning rates up to 15x base rate can cause training instability and NaN values. The system frequently enters collapse recovery mode due to these aggressive rates.

## Impact
- Frequent training crashes and poor convergence
- NaN values appearing in gradients and losses
- VAE collapse detection triggers too frequently
- System instability during real-time processing

## Evidence
```python
# Current aggressive rates in UltraFast60FpsLeanVAE.__init__()
self.instant_optimizer = optim.SGD(..., lr=learning_rate * 15)  # 15x base rate!
self.ultra_fast_optimizer = optim.SGD(..., lr=learning_rate * 10)  # 10x base rate!
```

## Reproduction
1. Run real-time processing with motion
2. Observe frequent "COLLAPSE RECOVERY MODE" messages
3. Monitor for NaN detection warnings
4. Check gradient magnitudes during training

## Acceptance Criteria
- [ ] Stable training without frequent collapse detection
- [ ] Learning rates based on gradient magnitude analysis
- [ ] Adaptive learning rate scheduling implementation
- [ ] Gradient clipping properly tuned
- [ ] Reduced NaN occurrences to <1% of training steps

## Suggested Solutions
1. **Implement adaptive learning rate scheduling** based on loss stability
2. **Use conservative base rates** (1-3x instead of 15x)
3. **Add gradient magnitude monitoring** for rate adjustment
4. **Implement learning rate warmup** for stable initialization
5. **Use gradient clipping** with proper norm thresholds

## Related Issues
- #009 (VAE collapse detection false positives)
- #003 (Energy conservation affected by instability)

## Test Reference
- Monitor collapse detector activation frequency
- Check for NaN values in training logs

## Priority Justification
High priority because it affects core system stability and prevents reliable real-time operation.

## Labels
- high-priority
- training
- stability
- learning-rates
- nan-issues