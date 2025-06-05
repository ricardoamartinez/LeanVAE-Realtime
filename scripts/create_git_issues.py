#!/usr/bin/env python3
"""
Script to create git issues from documented issues in docs/issues.md
"""

import os
import re
from datetime import datetime

# Issue templates based on the documented issues
ISSUES = [
    {
        'number': '002',
        'title': 'Fixed 3-Channel Input Requirement',
        'priority': 'Medium',
        'type': 'Limitation',
        'component': 'MicroLeanVAE, PhysicsInformedWaveletTransform',
        'description': 'The system is hardcoded for 3-channel (RGB) inputs. Grayscale or other channel configurations fail.',
        'impact': '- Cannot process grayscale video, depth maps, or multi-spectral data\n- Limits application to RGB-only scenarios\n- Reduces flexibility for different data types',
        'labels': ['enhancement', 'medium-priority', 'input-formats']
    },
    {
        'number': '004',
        'title': 'Memory Inefficiency in Spatiotemporal Processing',
        'priority': 'Medium',
        'type': 'Performance',
        'component': 'PhysicsInformedWaveletTransform',
        'description': 'The 3D spatiotemporal wavelet transform creates many intermediate tensors, leading to high memory usage.',
        'impact': '- Limits batch size and resolution for real-time processing\n- High memory overhead for temporal sequences\n- Potential memory fragmentation',
        'labels': ['performance', 'memory', 'optimization']
    },
    {
        'number': '007',
        'title': 'Missing Comprehensive Error Handling',
        'priority': 'High',
        'type': 'Robustness',
        'component': 'All modules',
        'description': 'Limited error handling for edge cases like NaN values, memory errors, or invalid inputs.',
        'impact': '- System crashes instead of graceful degradation\n- Difficult to debug issues in production\n- Poor user experience',
        'labels': ['critical', 'error-handling', 'robustness']
    },
    {
        'number': '009',
        'title': 'VAE Collapse Detection False Positives',
        'priority': 'Medium',
        'type': 'Algorithm',
        'component': 'CollapseDetector',
        'description': 'Collapse detection is too sensitive, triggering recovery mode too frequently.',
        'impact': '- Disrupts normal training flow\n- Unnecessary recovery operations\n- Reduced training efficiency',
        'labels': ['algorithm', 'vae', 'false-positives']
    },
    {
        'number': '011',
        'title': 'Limited Monitoring and Logging',
        'priority': 'Medium',
        'type': 'Feature Gap',
        'component': 'All modules',
        'description': 'No integration with tensorboard, wandb, or comprehensive logging.',
        'impact': '- Difficult to debug and monitor training progress\n- No metric visualization\n- Poor development experience',
        'labels': ['logging', 'monitoring', 'debugging']
    }
]

def create_issue_file(issue):
    """Create a git issue file from issue dictionary"""
    
    filename = f".github/issues/{issue['number']}-{issue['title'].lower().replace(' ', '-').replace('/', '-')}.md"
    
    content = f"""# Issue #{issue['number']}: {issue['title']}

**Version:** v1.0.0  
**Priority:** {issue['priority']}  
**Type:** {issue['type']}  
**Component:** {issue['component']}  
**Created:** {datetime.now().strftime('%Y-%m-%d')}  

## Description
{issue['description']}

## Impact
{issue['impact']}

## Acceptance Criteria
- [ ] Issue fully resolved and tested
- [ ] No regression in existing functionality
- [ ] Documentation updated
- [ ] Tests added/updated

## Labels
{', '.join([f'- {label}' for label in issue['labels']])}

## Status
- [ ] Open
- [ ] In Progress
- [ ] Testing
- [ ] Closed

## References
- See docs/issues.md for full context
- Related to Active Inference Agent v1.0.0 architecture
"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Created issue #{issue['number']}: {issue['title']}")

def main():
    """Create all git issues"""
    print("Creating git issues for Active Inference Agent v1.0.0...")
    
    for issue in ISSUES:
        create_issue_file(issue)
    
    print(f"\nCreated {len(ISSUES)} additional git issues!")
    print("Issues are now tracked in .github/issues/ directory")

if __name__ == '__main__':
    main()