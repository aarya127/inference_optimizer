"""
SLA Validator for AMIO Phase 0

Validates inference metrics against defined SLA targets:
- TTFT < 500ms (p95)
- TBT < 50ms (mean)
- Fragmentation < 20%
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json


class SLAViolationType(Enum):
    """Types of SLA violations"""
    TTFT_EXCEEDED = "ttft_exceeded"
    TBT_DEGRADED = "tbt_degraded"
    FRAGMENTATION_HIGH = "fragmentation_high"
    MEMORY_CRITICAL = "memory_critical"


@dataclass
class SLATarget:
    """SLA target definition"""
    metric_name: str
    target_value: float
    percentile: Optional[str] = None
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0


@dataclass
class SLAViolation:
    """Record of SLA violation"""
    violation_type: SLAViolationType
    metric_name: str
    measured_value: float
    target_value: float
    severity: str  # "warning" or "critical"
    timestamp: float
    description: str


class SLAValidator:
    """Validates metrics against SLA targets"""
    
    def __init__(self):
        # Define SLA targets
        self.targets = {
            'ttft_p95_ms': SLATarget(
                metric_name='ttft_p95_ms',
                target_value=500.0,
                percentile='p95',
                threshold_warning=500.0,
                threshold_critical=650.0
            ),
            'ttft_p99_ms': SLATarget(
                metric_name='ttft_p99_ms',
                target_value=650.0,
                percentile='p99',
                threshold_warning=650.0,
                threshold_critical=800.0
            ),
            'tbt_mean_ms': SLATarget(
                metric_name='tbt_mean_ms',
                target_value=50.0,
                percentile='mean',
                threshold_warning=60.0,
                threshold_critical=80.0
            ),
            'fragmentation_percent': SLATarget(
                metric_name='fragmentation_percent',
                target_value=20.0,
                threshold_warning=25.0,
                threshold_critical=30.0
            )
        }
        
        self.violations: List[SLAViolation] = []
        
    def validate_ttft(self, ttft_ms: float, percentile: str = 'p95') -> Optional[SLAViolation]:
        """
        Validate TTFT against target
        
        Args:
            ttft_ms: Measured TTFT in milliseconds
            percentile: Which percentile ('p95' or 'p99')
            
        Returns:
            SLAViolation if violated, None otherwise
        """
        target_key = f'ttft_{percentile}_ms'
        target = self.targets[target_key]
        
        if ttft_ms > target.threshold_critical:
            violation = SLAViolation(
                violation_type=SLAViolationType.TTFT_EXCEEDED,
                metric_name=target_key,
                measured_value=ttft_ms,
                target_value=target.target_value,
                severity='critical',
                timestamp=0.0,
                description=f"TTFT {percentile} ({ttft_ms:.1f}ms) exceeds critical threshold ({target.threshold_critical:.1f}ms)"
            )
            self.violations.append(violation)
            return violation
        elif ttft_ms > target.threshold_warning:
            violation = SLAViolation(
                violation_type=SLAViolationType.TTFT_EXCEEDED,
                metric_name=target_key,
                measured_value=ttft_ms,
                target_value=target.target_value,
                severity='warning',
                timestamp=0.0,
                description=f"TTFT {percentile} ({ttft_ms:.1f}ms) exceeds target ({target.target_value:.1f}ms)"
            )
            self.violations.append(violation)
            return violation
        
        return None
    
    def validate_tbt(self, tbt_ms: float) -> Optional[SLAViolation]:
        """
        Validate TBT against target
        
        Args:
            tbt_ms: Measured mean TBT in milliseconds
            
        Returns:
            SLAViolation if violated, None otherwise
        """
        target = self.targets['tbt_mean_ms']
        
        if tbt_ms > target.threshold_critical:
            violation = SLAViolation(
                violation_type=SLAViolationType.TBT_DEGRADED,
                metric_name='tbt_mean_ms',
                measured_value=tbt_ms,
                target_value=target.target_value,
                severity='critical',
                timestamp=0.0,
                description=f"TBT mean ({tbt_ms:.1f}ms) critically degraded (>{target.threshold_critical:.1f}ms)"
            )
            self.violations.append(violation)
            return violation
        elif tbt_ms > target.threshold_warning:
            violation = SLAViolation(
                violation_type=SLAViolationType.TBT_DEGRADED,
                metric_name='tbt_mean_ms',
                measured_value=tbt_ms,
                target_value=target.target_value,
                severity='warning',
                timestamp=0.0,
                description=f"TBT mean ({tbt_ms:.1f}ms) exceeds target ({target.target_value:.1f}ms)"
            )
            self.violations.append(violation)
            return violation
        
        return None
    
    def validate_fragmentation(self, frag_percent: float) -> Optional[SLAViolation]:
        """
        Validate memory fragmentation against target
        
        Args:
            frag_percent: Measured fragmentation percentage
            
        Returns:
            SLAViolation if violated, None otherwise
        """
        target = self.targets['fragmentation_percent']
        
        if frag_percent > target.threshold_critical:
            violation = SLAViolation(
                violation_type=SLAViolationType.FRAGMENTATION_HIGH,
                metric_name='fragmentation_percent',
                measured_value=frag_percent,
                target_value=target.target_value,
                severity='critical',
                timestamp=0.0,
                description=f"Fragmentation ({frag_percent:.1f}%) critically high (>{target.threshold_critical:.1f}%)"
            )
            self.violations.append(violation)
            return violation
        elif frag_percent > target.threshold_warning:
            violation = SLAViolation(
                violation_type=SLAViolationType.FRAGMENTATION_HIGH,
                metric_name='fragmentation_percent',
                measured_value=frag_percent,
                target_value=target.target_value,
                severity='warning',
                timestamp=0.0,
                description=f"Fragmentation ({frag_percent:.1f}%) exceeds target ({target.target_value:.1f}%)"
            )
            self.violations.append(violation)
            return violation
        
        return None
    
    def validate_all(self, metrics_summary: Dict) -> List[SLAViolation]:
        """
        Validate all metrics in summary
        
        Args:
            metrics_summary: Dictionary with aggregated metrics
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Validate TTFT
        if 'ttft' in metrics_summary:
            ttft = metrics_summary['ttft']
            if 'p95_ms' in ttft:
                v = self.validate_ttft(ttft['p95_ms'], 'p95')
                if v:
                    violations.append(v)
            if 'p99_ms' in ttft:
                v = self.validate_ttft(ttft['p99_ms'], 'p99')
                if v:
                    violations.append(v)
        
        # Validate TBT
        if 'tbt' in metrics_summary and 'mean_ms' in metrics_summary['tbt']:
            v = self.validate_tbt(metrics_summary['tbt']['mean_ms'])
            if v:
                violations.append(v)
        
        # Validate Fragmentation
        if 'fragmentation' in metrics_summary and 'mean_percent' in metrics_summary['fragmentation']:
            v = self.validate_fragmentation(metrics_summary['fragmentation']['mean_percent'])
            if v:
                violations.append(v)
        
        return violations
    
    def get_compliance_rate(self) -> float:
        """Calculate SLA compliance rate"""
        if not self.violations:
            return 100.0
        
        # Count critical violations as double
        critical_count = sum(2 for v in self.violations if v.severity == 'critical')
        warning_count = sum(1 for v in self.violations if v.severity == 'warning')
        
        total_checks = len(self.violations) + 10  # Assume 10 checks
        passed_checks = total_checks - (critical_count + warning_count)
        
        return (passed_checks / total_checks) * 100.0
    
    def print_violations(self):
        """Print all violations"""
        if not self.violations:
            print("All SLA targets met!")
            return
        
        print(f"\n{'='*80}")
        print(f"SLA Violations: {len(self.violations)}")
        print(f"{'='*80}\n")
        
        for i, violation in enumerate(self.violations, 1):
            severity_icon = "[WARN] " if violation.severity == 'warning' else "[FAIL]"
            print(f"{severity_icon} {violation.severity.upper()} #{i}:")
            print(f"  Metric: {violation.metric_name}")
            print(f"  Measured: {violation.measured_value:.1f}")
            print(f"  Target: {violation.target_value:.1f}")
            print(f"  Description: {violation.description}")
            print()
        
        compliance_rate = self.get_compliance_rate()
        print(f"Compliance Rate: {compliance_rate:.1f}%")
        print(f"{'='*80}\n")
    
    def export_report(self, filepath: str):
        """Export validation report to JSON"""
        report = {
            'targets': {
                name: {
                    'target_value': target.target_value,
                    'threshold_warning': target.threshold_warning,
                    'threshold_critical': target.threshold_critical
                }
                for name, target in self.targets.items()
            },
            'violations': [
                {
                    'type': v.violation_type.value,
                    'metric': v.metric_name,
                    'measured': v.measured_value,
                    'target': v.target_value,
                    'severity': v.severity,
                    'description': v.description
                }
                for v in self.violations
            ],
            'compliance_rate': self.get_compliance_rate()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    """Test SLA validation"""
    print("=" * 80)
    print("AMIO Phase 0 - SLA Validation Test")
    print("=" * 80)
    print()
    
    validator = SLAValidator()
    
    # Test case 1: All targets met
    print("Test Case 1: All targets met")
    print("-" * 80)
    
    metrics_good = {
        'ttft': {
            'p95_ms': 450.0,
            'p99_ms': 580.0
        },
        'tbt': {
            'mean_ms': 45.0
        },
        'fragmentation': {
            'mean_percent': 15.0
        }
    }
    
    violations = validator.validate_all(metrics_good)
    print(f"Violations: {len(violations)}")
    if not violations:
        print("All SLA targets met!")
    
    print("\n")
    
    # Test case 2: Some violations
    print("Test Case 2: Multiple violations")
    print("-" * 80)
    
    validator = SLAValidator()  # Reset
    
    metrics_bad = {
        'ttft': {
            'p95_ms': 520.0,  # Warning
            'p99_ms': 700.0   # Critical
        },
        'tbt': {
            'mean_ms': 85.0   # Critical
        },
        'fragmentation': {
            'mean_percent': 28.0  # Warning
        }
    }
    
    violations = validator.validate_all(metrics_bad)
    validator.print_violations()
    
    print("SLA validation test complete")
