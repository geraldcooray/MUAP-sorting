"""Shared spike-detection, feature-extraction, clustering, metrics, and
plotting building blocks for the spike_sort_emg.py entry-point script.

The entry-point script keeps its own tunable-parameters block and argparse
CLI, and wires these modules together in its own main(). Nothing in this
package reads that script's constants directly -- every function takes its
parameters explicitly, so each module is independently importable/testable.
"""
