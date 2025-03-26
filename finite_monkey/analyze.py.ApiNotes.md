The comprehensive pipeline now retrieves the query engine from the PipelineFactory’s cached instance.
The new method get_query_engine() in PipelineFactory guarantees the FLAREInstructQueryEngine is instantiated only once, enhancing both performance and consistency.
Integrating the query engine into the context (if necessary) can further allow stages to access uniform, structured query capabilities.
