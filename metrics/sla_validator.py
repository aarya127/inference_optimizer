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
    """Validates metrics against SLA targets.

    Bookkeeping:
    - `checks_performed` counts every validate_* invocation, so the
      compliance rate has a real denominator (a previous version invented
      one — `len(violations) + 10` — which double-weighted criticals and
      could go negative).
    - Warnings trigger AT the target value: threshold_warning == target for
      every SLATarget below, so a measurement between the target and the
      critical threshold is reported instead of silently passing.
    - `validate_all` resets accumulated violations/checks at the start, so
      reusing one validator instance does not double-count.
    """

    def __init__(self):
        # Define SLA targets. threshold_warning == target_value by design:
        # anything over target at least warns (no hidden margin).
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
                threshold_warning=50.0,
                threshold_critical=80.0
            ),
            'fragmentation_percent': SLATarget(
                metric_name='fragmentation_percent',
                target_value=20.0,
                threshold_warning=20.0,
                threshold_critical=30.0
            )
        }

        self.violations: List[SLAViolation] = []
        self.checks_performed: int = 0

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
        self.checks_performed += 1

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
        self.checks_performed += 1

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
        self.checks_performed += 1

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
        # Reset accumulated state so reusing the same validator instance
        # does not double-count violations or checks across calls.
        self.violations = []
        self.checks_performed = 0

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
        """SLA compliance rate in [0, 1].

        Defined as 1 − violations/checks over the REAL number of checks
        performed by validate_* calls (self.checks_performed).  Each check
        produces at most one violation, so the result is always in [0, 1].
        Severity does not affect this rate — see
        get_severity_weighted_score() for a weighted view.

        Returns 1.0 when no checks have been performed yet.
        """
        if self.checks_performed == 0:
            return 1.0
        return 1.0 - len(self.violations) / self.checks_performed

    def get_severity_weighted_score(self, critical_weight: float = 2.0) -> float:
        """Severity-weighted penalty score in [0, 1] (1.0 = clean).

        This is NOT the compliance rate: criticals are weighted
        `critical_weight`× relative to warnings, and the result is clamped
        at 0.  Use get_compliance_rate() for the plain violations/checks
        fraction.
        """
        if self.checks_performed == 0:
            return 1.0
        penalty = sum(
            critical_weight if v.severity == 'critical' else 1.0
            for v in self.violations
        )
        return max(0.0, 1.0 - penalty / (self.checks_performed * critical_weight))
    
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
        print(f"Compliance Rate: {compliance_rate * 100:.1f}%  "
              f"({len(self.violations)} violations / "
              f"{self.checks_performed} checks)")
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
            'checks_performed': self.checks_performed,
            'compliance_rate': self.get_compliance_rate(),          # 0..1
            'severity_weighted_score': self.get_severity_weighted_score(),  # 0..1
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
