import streamlit as st
import asyncio
from typing import Dict, Any
from pathlib import Path
from finite_monkey.pipeline.factory import PipelineFactory
from finite_monkey.pipeline.core import Context

class StreamlitApp:
    def __init__(self):
        self.pipeline_factory = PipelineFactory()
        if 'debug' not in st.session_state:
            st.session_state.debug = False
        
    def run(self):
        st.title("Finite Monkey Engine")
        
        # Debug toggle in sidebar
        st.sidebar.checkbox("Debug Mode", key='debug')
        
        # Sidebar configuration
        st.sidebar.header("Configuration")
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["Single File", "Directory Scan"]
        )
        
        # Main content
        if analysis_type == "Single File":
            self._run_single_file_analysis()
        else:
            self._run_directory_analysis()
    
    def _run_single_file_analysis(self):
        uploaded_file = st.file_uploader("Choose a Solidity file", type=['sol'])
        
        if uploaded_file:
            # Save uploaded file
            temp_path = Path("temp") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(uploaded_file.getvalue())
            
            # Configure analysis
            output_path = st.text_input(
                "Output Path",
                value=str(Path("output") / f"{uploaded_file.name}_analysis.html")
            )
            
            config = {
                'chunk_size': st.slider("Chunk Size", 500, 2000, 1000),
                'overlap': st.slider("Overlap", 50, 200, 100),
                'report_format': st.selectbox("Report Format", ['html', 'json', 'markdown'])
            }
            
            if st.button("Analyze"):
                with st.spinner("Running analysis..."):
                    # Create and run pipeline
                    pipeline = self.pipeline_factory.create_standard_pipeline(
                        input_paths=[str(temp_path)],
                        output_path=output_path,
                        config=config
                    )
                    
                    context = asyncio.run(self._run_pipeline(pipeline))
                    
                    # Show results
                    self._display_results(context)
    
    def _run_directory_analysis(self):
        directory = st.text_input("Directory Path")
        
        if directory and Path(directory).exists():
            output_path = st.text_input(
                "Output Path",
                value=str(Path("output") / "batch_analysis.html")
            )
            
            config = {
                'chunk_size': st.slider("Chunk Size", 500, 2000, 1000),
                'overlap': st.slider("Overlap", 50, 200, 100),
                'report_format': st.selectbox("Report Format", ['html', 'json', 'markdown'])
            }
            
            if st.button("Analyze Directory"):
                with st.spinner("Running batch analysis..."):
                    pipeline = self.pipeline_factory.create_batch_pipeline(
                        directory=directory,
                        output_path=output_path,
                        config=config
                    )
                    
                    context = asyncio.run(self._run_pipeline(pipeline))
                    
                    # Show results
                    self._display_results(context)
    
    async def _run_pipeline(self, pipeline) -> Context:
        """Run pipeline and return context"""
        try:
            context = Context()
            if st.session_state.debug:
                st.write("Starting pipeline execution...")
            
            result = await pipeline.run(context)
            
            if st.session_state.debug:
                st.write("Pipeline execution completed")
            
            return result
            
        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")
            if st.session_state.debug:
                st.exception(e)
            raise
    
    def _display_results(self, context: Context):
        """Display analysis results in Streamlit"""
        st.header("Analysis Results")
        
        # Display metrics
        if context.metrics:
            st.subheader("Metrics")
            cols = st.columns(len(context.metrics))
            for col, (key, value) in zip(cols, context.metrics.items()):
                col.metric(key, value)
        
        # Display findings
        if context.findings:
            st.subheader("Findings")
            for finding in context.findings:
                with st.expander(f"{finding.type} ({finding.severity})"):
                    st.markdown(f"**Location:** {finding.location}")
                    st.markdown(f"**Description:** {finding.description}")
                    if hasattr(finding, 'recommendation'):
                        st.markdown(f"**Recommendation:** {finding.recommendation}")
        
        # Display errors if any
        if context.errors:
            st.subheader("Errors")
            for error in context.errors:
                st.error(f"{error.source}: {error.message}")
        
        # Show report link if generated
        if "report_path" in context.state:
            st.success(f"Report generated at: {context.state['report_path']}")
            if Path(context.state['report_path']).exists():
                with open(context.state['report_path'], 'r') as f:
                    st.download_button(
                        "Download Report",
                        f.read(),
                        file_name=Path(context.state['report_path']).name
                    )

def main():
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()