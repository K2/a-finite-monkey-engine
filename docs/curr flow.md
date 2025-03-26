```mermaid
graph TD
    A[Source Input] --> B[Chunking]
    B --> C[Analysis]
    C --> D[Validation]
    D --> E[Documentation]
    
    subgraph Database Tracking
        PT[Project Tasks]
        AR[Analysis Results]
        PR[Prompts Registry]
    end
    
    B -- Track Progress --> PT
    C -- Store Findings --> AR
    D -- Update Validation --> AR
    PR -- Feed Prompts --> C
    style A fill:#3498db,stroke:#333,stroke-width:2px,color:black
    style B fill:#2ecc71,stroke:#333,stroke-width:2px,color:black
    style C fill:#9b59b6,stroke:#333,stroke-width:2px,color:black
    style D fill:#f1c40f,stroke:#333,stroke-width:2px,color:black
    style E fill:#e67e22,stroke:#333,stroke-width:2px,color:black
    style PT fill:#00bcd4,stroke:#333,stroke-width:2px,color:black
    style AR fill:#ff80ab,stroke:#333,stroke-width:2px,color:black
    style PR fill:#8bc34a,stroke:#333,stroke-width:2px,color:black
```

```sql
        CREATE TABLE project_tasks (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR NOT NULL,
        task_type VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        result JSONB,
        metadata JSONB
    );


        CREATE TABLE analysis_results (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR NOT NULL,
        file_path VARCHAR NOT NULL,
        chunk_id VARCHAR,
        finding_type VARCHAR NOT NULL,
        severity VARCHAR NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        validated BOOLEAN DEFAULT FALSE,
        validation_result JSONB
    );


        CREATE TABLE prompts_registry (
        id SERIAL PRIMARY KEY,
        context_type VARCHAR NOT NULL,
        language VARCHAR NOT NULL,
        prompt_category VARCHAR NOT NULL,
        prompt_text TEXT NOT NULL,
        parameters JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used TIMESTAMP,
        success_rate FLOAT DEFAULT 0.0,
        metadata JSONB
    );
```

# Flow Phases
* Source Input

    * File/Directory/GitHub input
    * Initial project task creation
    * Metadata collection
    * Chunking

    * Code splitting
    * Context preservation
    * Progress tracking
    * Analysis

    * Prompt selection from registry
    * LLM interaction
    * Finding generation
    * Result storage
    * Validation

    * Result verification
    * Cross-reference checking
    * Confidence scoring
    * Documentation

    * Report generation
    * Evidence collection
    * Result summarization

# Database Interactions

```mermaid
sequenceDiagram
    participant P as 🔷 Pipeline
    participant PT as 🟢 ProjectTasks
    participant AR as 🟡 AnalysisResults
    participant PR as 🟣 PromptsRegistry
    
    activate P
    P->>PT: Create new task
    activate PT
    deactivate PT
    
    P->>PR: Get relevant prompts
    activate PR
    PR-->>P: Return context-specific prompts
    deactivate PR
    
    P->>AR: Store findings
    activate AR
    deactivate AR
    
    P->>AR: Update with validation
    activate AR
    deactivate AR
    
    P->>PT: Mark task complete
    activate PT
    deactivate PT
    deactivate P
```

```sql
    SELECT prompt_text, parameters 
    FROM prompts_registry 
    WHERE context_type = :context_type 
      AND language = :language 
      AND prompt_category = :category
    ORDER BY success_rate DESC 
    LIMIT 1;

    INSERT INTO project_tasks 
    (project_id, task_type, status, metadata)
    VALUES (:project_id, :task_type, 'started', :metadata);
    INSERT INTO analysis_results 
    (project_id, file_path, finding_type, severity, description)
    VALUES (:project_id, :file_path, :finding_type, :severity, :description);
```


* This structure provides a complete feedback loop where
* Each code context gets appropriate prompts
* Results are tracked and validated
* Prompt effectiveness is measured
* Pipeline progress is monitored