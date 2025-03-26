# Analysis Flow Chart

This chart illustrates how information flows through the different analyzers and how findings are shared between components.

```mermaid
flowchart TB
    subgraph Input
        A1[Local Files]
        A2[GitHub Repository]
    end
    
    subgraph Processing
        B[Contract Chunker]
        C[Function Extractor]
        
        subgraph "Primary Analysis"
            D[Business Flow Analyzer]
            E[Data Flow Analyzer]
            F[Vulnerability Scanner]
        end
        
        subgraph "Enhanced Analysis"
            G[Cognitive Bias Analyzer]
            H[Documentation Analyzer]
            I[Doc Inconsistency Analyzer]
            J[Counterfactual Analyzer]
        end
        
        K[Flow Joiner]
    end
    
    subgraph Output
        L[Report Generator]
        M[HTML Report]
        N[JSON Findings]
    end
    
    %% Main flow
    A1 --> B
    A2 --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    L --> N
    
    %% Cross-cutting flows
    D -- "Business flows" --> E
    D -- "Business context" --> G
    D -- "Business context" --> J
    
    E -- "Dataflows" --> J
    E -- "Vulnerable paths" --> F
    
    F -- "Security findings" --> H
    F -- "Security findings" --> I
    
    G -- "Cognitive biases" --> E
    G -- "Developer assumptions" --> J
    
    style G fill:#f9d5e5,stroke:#333,stroke-width:2px
    style E fill:#d5e8d4,stroke:#333,stroke-width:2px
    style J fill:#ffe6cc,stroke:#333,stroke-width:2px
    style I fill:#e1d5e7,stroke:#333,stroke-width:2px
    
    classDef newAnalyzer fill:#ffe6cc,stroke:#333,stroke-width:2px
    class G,H,I,J newAnalyzer
```

## Information Sharing Between Components

Each analyzer shares its findings with other analyzers to provide deeper insights:

1. **Business Flow Analyzer** → **Data Flow Analyzer**
   - Provides business context for data flows
   - Helps identify high-impact paths

2. **Data Flow Analyzer** → **Counterfactual Analyzer**
   - Shares vulnerable data paths
   - Informs "what if" scenarios

3. **Cognitive Bias Analyzer** → **Data Flow Analyzer**
   - Shares identified biases affecting code paths
   - Helps assess exploitability

4. **Vulnerability Scanner** → **Documentation Analyzer**
   - Provides security context for documentation evaluation
   - Highlights under-documented vulnerable areas

5. **All Analyzers** → **Report Generator**
   - Comprehensive findings integration
   - Cross-referenced security insights
