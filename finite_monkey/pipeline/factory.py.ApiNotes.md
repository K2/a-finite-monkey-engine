The following changes were made:
1. Added a helper function (require_attributes) to check context attributes for required data.
2. Cached instances of ThreatDetector, CounterfactualAnalyzer, CognitiveBiasAnalyzer, DocumentationAnalyzer, and DocumentationInconsistencyAdapter built during factory initialization.
3. Refactored each stage function to use require_attributes() for early exits.
4. Minor performance and consistency improvements.
