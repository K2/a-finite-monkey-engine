# Adaptive Query Threshold Visualization

## Design Philosophy

The adaptive query system addresses a key limitation of fixed top-k retrieval by dynamically determining the optimal number of documents to retrieve based on similarity score patterns. This approach follows several key principles:

1. **Relevance-Based Cutoff**: Only documents with meaningful semantic similarity are included
2. **Significant Drop-off Detection**: Identifies natural breaking points in the similarity score distribution
3. **Minimum Context Guarantees**: Always includes a minimum number of documents to ensure sufficient context
4. **Maximum Size Control**: Prevents context explosion by setting an upper bound on retrievals

## Technical Implementation

The implementation uses a multi-factor approach to determine where to "cut off" document retrieval:

### Similarity Score Analysis
- **Absolute Threshold**: Documents below a minimum similarity threshold are excluded
- **Relative Drop-off**: Detects when scores decline significantly relative to the highest score
- **Sequential Drop-off**: Identifies sharp declines between consecutive document scores

### Visualization Components
1. **Score Distribution Chart**: Displays similarity scores in descending order
2. **Cutoff Indicators**: Visual markers showing:
   - Minimum document count (min_k)
   - Similarity threshold line
   - Adaptive cutoff point
   - Drop-off annotations

3. **Selection Highlighting**: Clear distinction between selected and excluded documents

## Parameter Tuning

The visualization tool helps tune four critical parameters:

1. `min_k`: Minimum documents to always include (context floor)
2. `max_k`: Maximum documents to consider (context ceiling)
3. `similarity_threshold`: Absolute minimum similarity score
4. `drop_off_factor`: Sensitivity to similarity decline (0.0-1.0)

By visualizing how these parameters affect document selection across different queries, you can fine-tune the system to your specific corpus and query patterns.

## Usage Recommendations

- Start with conservative parameters: min_k=5, max_k=30, threshold=0.1, drop_off=0.5
- Run varied queries representative of your use cases
- Examine where natural drop-offs occur in your similarity distributions
- Adjust parameters to match the observed patterns
- Consider different parameter sets for different query types
