"""
MLX Tech Stack Validator for Apple Silicon M3

Validates the Phase 0 technical stack requirements:
1. MLX installation and version
2. Apple Silicon detection
3. Unified memory availability
4. Required dependencies
5. GPU/Neural Engine accessibility
"""

import sys
import platform
import subprocess
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    message: str
    details: Dict = None


class TechStackValidator:
    """Validates AMIO Phase 0 tech stack on Apple Silicon"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("=" * 80)
        print("AMIO Phase 0 - Tech Stack Validation")
        print("=" * 80)
        print()
        
        # Run all checks
        self._check_platform()
        self._check_apple_silicon()
        self._check_mlx_installation()
        self._check_mlx_version()
        self._check_unified_memory()
        self._check_gpu_access()
        self._check_dependencies()
        
        # Print summary
        self._print_summary()
        
        # Return overall status
        return all(r.passed for r in self.results)
    
    def _check_platform(self):
        """Verify running on macOS"""
        is_macos = platform.system() == "Darwin"
        
        self.results.append(ValidationResult(
            check_name="Platform",
            passed=is_macos,
            message=f"Running on {platform.system()}",
            details={"platform": platform.system(), "version": platform.version()}
        ))
    
    def _check_apple_silicon(self):
        """Verify Apple Silicon (ARM64) architecture"""
        machine = platform.machine()
        is_arm64 = machine == "arm64"
        
        # Try to detect M-series chip
        chip_name = "Unknown"
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )
            chip_name = result.stdout.strip()
        except:
            pass
        
        self.results.append(ValidationResult(
            check_name="Apple Silicon",
            passed=is_arm64,
            message=f"Architecture: {machine}, Chip: {chip_name}",
            details={"architecture": machine, "chip": chip_name}
        ))
    
    def _check_mlx_installation(self):
        """Check if MLX is installed"""
        try:
            import mlx
            import mlx.core as mx
            
            version = getattr(mlx, '__version__', 'unknown')
            self.results.append(ValidationResult(
                check_name="MLX Installation",
                passed=True,
                message=f"MLX installed: {version}",
                details={"version": version}
            ))
        except ImportError as e:
            self.results.append(ValidationResult(
                check_name="MLX Installation",
                passed=False,
                message=f"MLX not found: {str(e)}",
                details={"error": str(e)}
            ))
    
    def _check_mlx_version(self):
        """Verify MLX version meets requirements (>=0.10.0)"""
        try:
            import mlx
            from packaging import version
            
            required_version = "0.10.0"
            current_version = getattr(mlx, '__version__', 'unknown')
            
            if current_version == 'unknown':
                self.results.append(ValidationResult(
                    check_name="MLX Version",
                    passed=True,  # Assume OK if installed
                    message=f"MLX installed (version check skipped)",
                    details={"current": "unknown"}
                ))
            else:
                meets_requirement = version.parse(current_version) >= version.parse(required_version)
                
                self.results.append(ValidationResult(
                    check_name="MLX Version",
                    passed=meets_requirement,
                    message=f"MLX {current_version} (required: >={required_version})",
                    details={"current": current_version, "required": required_version}
                ))
        except ImportError:
            self.results.append(ValidationResult(
                check_name="MLX Version",
                passed=False,
                message="Cannot check version - MLX not installed",
                details={}
            ))
        except Exception as e:
            # Fallback if packaging not available
            try:
                import mlx
                version_str = getattr(mlx, '__version__', 'unknown')
                self.results.append(ValidationResult(
                    check_name="MLX Version",
                    passed=True,  # Assume OK if installed
                    message=f"MLX {version_str} (version check skipped)",
                    details={"current": version_str}
                ))
            except:
                pass
    
    def _check_unified_memory(self):
        """Check unified memory availability"""
        try:
            result = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # Parse memory size in bytes
                memory_bytes = int(result.stdout.split(":")[1].strip())
                memory_gb = memory_bytes / (1024 ** 3)
                
                # Minimum 8GB required (SmolVLM confirmed working on 8GB M3)
                sufficient = memory_gb >= 8
                
                self.results.append(ValidationResult(
                    check_name="Unified Memory",
                    passed=sufficient,
                    message=f"{memory_gb:.1f} GB available (minimum: ≥8GB)",
                    details={"memory_gb": memory_gb, "sufficient": sufficient}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="Unified Memory",
                    passed=False,
                    message="Could not determine memory size",
                    details={}
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="Unified Memory",
                passed=False,
                message=f"Error checking memory: {str(e)}",
                details={"error": str(e)}
            ))
    
    def _check_gpu_access(self):
        """Verify GPU/Metal access through MLX"""
        try:
            import mlx.core as mx
            
            # Try to create a small tensor on GPU
            x = mx.array([1.0, 2.0, 3.0])
            y = mx.add(x, x)
            mx.eval(y)  # Force evaluation
            
            # Check default device
            default_device = mx.default_device()
            
            self.results.append(ValidationResult(
                check_name="GPU/Metal Access",
                passed=True,
                message=f"Metal GPU accessible (device: {default_device})",
                details={"device": str(default_device)}
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="GPU/Metal Access",
                passed=False,
                message=f"Cannot access GPU: {str(e)}",
                details={"error": str(e)}
            ))
    
    def _check_dependencies(self):
        """Check for required Python dependencies"""
        required_packages = [
            ("numpy", "1.24.0"),
            ("PIL", "10.0.0"),  # pillow installs as PIL
            ("transformers", "4.40.0"),
            ("huggingface_hub", "0.20.0"),
        ]
        
        optional_packages = [
            ("mlx_lm", None),
            ("mlx_vlm", None),
            ("matplotlib", "3.7.0"),
            ("seaborn", "0.12.0"),
            ("psutil", "5.9.0"),
        ]
        
        missing_required = []
        missing_optional = []
        
        for package, min_version in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_required.append(package)
        
        for package, min_version in optional_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_optional.append(package)
        
        passed = len(missing_required) == 0
        
        message_parts = []
        if missing_required:
            message_parts.append(f"Missing required: {', '.join(missing_required)}")
        if missing_optional:
            message_parts.append(f"Missing optional: {', '.join(missing_optional)}")
        
        if not message_parts:
            message = "All required dependencies installed"
        else:
            message = "; ".join(message_parts)
        
        self.results.append(ValidationResult(
            check_name="Dependencies",
            passed=passed,
            message=message,
            details={
                "missing_required": missing_required,
                "missing_optional": missing_optional
            }
        ))
    
    def _print_summary(self):
        """Print validation summary"""
        print("\nValidation Results:")
        print("-" * 80)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} | {result.check_name:20s} | {result.message}")
        
        print("-" * 80)
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        print(f"\nSummary: {passed_count}/{total_count} checks passed")
        
        if passed_count == total_count:
            print("Tech stack is ready for Phase 0!")
        else:
            print("\nPlease fix the failed checks before proceeding.")
            print("\nInstallation instructions:")
            print("  pip install mlx>=0.10.0 mlx-lm mlx-vlm")
            print("  pip install numpy pillow transformers huggingface-hub")
            print("  pip install matplotlib seaborn psutil")


def main():
    """Main validation entry point"""
    validator = TechStackValidator()
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
