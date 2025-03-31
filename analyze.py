"""
Entry point for the Finite Monkey Engine analysis pipeline.

This file configures and executes the pipeline. All stages use a common interface,
including a dedicated query engine stage for structure-aware processing.
"""
import asyncio
import logging
from finite_monkey.nodes_config import config
from finite_monkey.core.context import Context
from finite_monkey.pipeline.factory import PipelineFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_comprehensive_pipeline(factory: PipelineFactory, options: Dict[str, Any]) -> Pipeline:
    # Map stage names to their corresponding methods (factory stage builders)
    stage_methods = {}
    all_stages = [
        "load_project",
        "load_documents",
        "chunk_documents",
        "extract_business_flows",
        "_analyze_vector_prompts_stage",  # Assuming this is a stage method
        "threat_detection_stage",
        "counterfactual_stage",
        "cognitive_bias_stage",
        "documentation_analysis_stage",
        "analyze_documentation_inconsistency_stage"
    ]
    for stage in all_stages:
        try:
            factory_method = getattr(factory, stage, None)
            if factory_method and callable(factory_method):
                stage_methods[stage] = factory_method()  # invoke stage builder
            else:
                logger.warning(f"Optional stage not available: {stage}")
        except Exception as e:
            logger.warning(f"Error checking stage {stage}: {e}")
    
    available_stages = list(stage_methods.keys())
    if options.get("stages"):
        selected_stages = [s for s in options["stages"] if s in available_stages]
    else:
        selected_stages = available_stages
    if options.get("exclude_stages"):
        selected_stages = [s for s in selected_stages if s not in options["exclude_stages"]]
    
    pipeline_stages = []
    for stage_name in selected_stages:
        try:
            if stage_name in stage_methods:
                stage_method = stage_methods[stage_name]
                pipeline_stages.append(stage_method)
                logger.debug(f"Added stage to pipeline: {stage_name}")
            else:
                logger.error(f"Stage method not found for {stage_name}")
        except Exception as e:
            logger.error(f"Error adding stage {stage_name}: {e}")
    
    logger.info(f"Creating pipeline with {len(pipeline_stages)} stages")
    # Set the query engine by retrieving the cached instance from the factory
    query_engine = factory.get_query_engine()
    # Optional: Pass the query engine into the pipeline context if needed
    # For example, context.query_engine = query_engine (ensure Context supports this)
    
    pipeline = Pipeline(stages=pipeline_stages)
    return pipeline

async def main():
    context = Context(input_path="path/to/your/project")
    factory = PipelineFactory(config)
    # Create the comprehensive pipeline with helper options
    options = {}  # Adjust options as needed
    pipeline = await create_comprehensive_pipeline(factory, options)
    for stage in pipeline.stages:
        logger.info(f"Executing stage: {stage.__name__ if hasattr(stage, '__name__') else stage}")
        context = await stage(context)
    logger.info("Pipeline execution completed")
    logger.info(f"Final Context: {context}")
    
if __name__ == "__main__":
    asyncio.run(main())
