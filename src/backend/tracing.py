"""
Consolidated tracing setup for Azure AI Foundry.

Uses Agent Framework's setup_observability for proper span nesting.
Sends traces to Azure Application Insights connected to Foundry project.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory
_backend_dir = Path(__file__).parent
_env_file = _backend_dir / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# Suppress noisy loggers
logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)
logging.getLogger("azure.monitor.opentelemetry.exporter.export").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

_TRACING_CONFIGURED = False


def _get_foundry_appinsights_connection_string() -> str:
    """
    Get Application Insights connection string from Foundry project.
    This ensures traces appear in Foundry portal's Tracing tab.
    """
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    if not project_endpoint:
        return None
    
    try:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        
        project_client = AIProjectClient(
            credential=DefaultAzureCredential(),
            endpoint=project_endpoint
        )
        
        conn_str = project_client.telemetry.get_application_insights_connection_string()
        print(f"📊 Got App Insights connection string from Foundry project")
        return conn_str
    except Exception as e:
        print(f"⚠️  Could not get connection string from Foundry: {e}")
        return None


def setup_tracing(
    connection_string: str = None,
    service_name: str = "moneta-agents"
) -> bool:
    """
    Configure tracing using Agent Framework's setup_observability.
    
    This ensures proper span nesting for:
    - invoke_agent spans
    - chat model spans  
    - execute_tool spans
    
    Call this ONCE at application startup.
    """
    global _TRACING_CONFIGURED
    
    if _TRACING_CONFIGURED:
        return True
    
    # Priority: 1) Provided connection string, 2) Foundry project's App Insights, 3) Env var
    conn_str = connection_string
    if not conn_str:
        conn_str = _get_foundry_appinsights_connection_string()
    if not conn_str:
        conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if not conn_str:
        print("⚠️  No Application Insights connection string available - tracing disabled")
        return False
    
    try:
        # Use Agent Framework's built-in observability setup
        # This properly instruments agent spans with parent-child relationships
        from agent_framework.observability import setup_observability
        
        setup_observability(
            enable_sensitive_data=True,  # Include prompts/responses in traces
            applicationinsights_connection_string=conn_str
        )
        
        _TRACING_CONFIGURED = True
        print(f"✅ Tracing configured via Agent Framework setup_observability")
        print(f"   - Application Insights: configured (Foundry-connected)")
        print(f"   - Sensitive data recording: enabled")
        return True
        
    except ImportError as e:
        print(f"⚠️  Agent Framework observability not available: {e}")
        print("   Falling back to manual OpenTelemetry setup...")
        return _setup_manual_tracing(conn_str, service_name)
    except Exception as e:
        print(f"❌ Agent Framework observability failed: {e}")
        print("   Falling back to manual OpenTelemetry setup...")
        return _setup_manual_tracing(conn_str, service_name)


def _setup_manual_tracing(conn_str: str, service_name: str) -> bool:
    """Fallback manual OpenTelemetry setup if Agent Framework is unavailable."""
    global _TRACING_CONFIGURED
    
    # Enable content recording for Gen AI traces
    os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.semconv.resource import ResourceAttributes
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        
        # Create resource with service name
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Add Azure Monitor exporter
        exporter = AzureMonitorTraceExporter(connection_string=conn_str)
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        
        # Set as global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        print(f"📊 Tracing: Azure Monitor exporter configured (manual)")
        
        # Enable Azure AI Inference instrumentor for Foundry Gen AI traces
        try:
            from azure.ai.inference.tracing import AIInferenceInstrumentor
            AIInferenceInstrumentor().instrument()
            print(f"📊 Tracing: AI Inference instrumentor enabled")
        except Exception as e:
            print(f"⚠️  AIInferenceInstrumentor: {e}")
        
        _TRACING_CONFIGURED = True
        print(f"✅ Tracing configured for App Insights (manual fallback)")
        return True
        
    except Exception as e:
        print(f"❌ Manual tracing configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_tracer(name: str = __name__):
    """Get an OpenTelemetry tracer for custom spans."""
    try:
        # Prefer Agent Framework's get_tracer if available
        from agent_framework.observability import get_tracer as af_get_tracer
        return af_get_tracer()
    except ImportError:
        from opentelemetry import trace
        return trace.get_tracer(name)


# Backward compatibility - simple tracing manager stub
class _TracingManager:
    """Simple tracing manager for backward compatibility."""
    
    def __init__(self):
        self.is_configured = True
        self._tracer = None
    
    @property
    def tracer(self):
        if self._tracer is None:
            self._tracer = get_tracer("function-tracing")
        return self._tracer
    
    def trace_function_call(self, function_name: str, **kwargs):
        """Create a span for a function call."""
        return self.tracer.start_as_current_span(function_name, attributes=kwargs)


_tracing_manager = None

def get_tracing_manager():
    """Get the global tracing manager instance."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = _TracingManager()
    return _tracing_manager
