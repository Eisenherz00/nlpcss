"""Terminal report printing for evaluation results."""


def print_report(report: dict) -> None:
    """Print a human-readable summary of an evaluate_batch report."""
    print(f"\n=== Evaluation Summary ({report['summary']['n_items']} items) ===\n")
    print("Implemented checks (rule-based):")
    for check_name, rate in report["summary"]["pass_rates_per_check"].items():
        bar = "█" * int(rate * 20)
        print(f"  {check_name:<35} {rate*100:5.1f}%  {bar}")
    print(f"\nMean pass rate (implemented): {report['summary']['mean_pass_rate_implemented']*100:.1f}%")
    print(f"\nDeferred checks (need LLM-judge): {report['summary']['deferred_checks']}")
