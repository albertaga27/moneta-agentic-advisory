"""
Shared tracing utilities for Azure AI Agents with OpenTelemetry integration.

This module provides consistent tracing setup across all banking agents,
supporting Azure AI Foundry tracing, Azure Monitor, and local console tracing.

IMPORTANT: For Azure AI Foundry, tracing is configured via:
1. APPLICATIONINSIGHTS_CONNECTION_STRING - for Azure Monitor export
2. azure.ai.inference.tracing.AIInferenceInstrumentor - for AI-specific telemetry

The tracing should be initialized ONCE at the orchestrator level, not per-agent.
"""

import os
import sys
import logging
import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Suppress the "Calling end() on an ended span" warning from OpenTelemetry
logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)

# OpenTelemetry imports
from azure.core.settings import settings
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

# Track if tracing is already configured globally (singleton pattern)
_TRACING_CONFIGURED = False
_tracing_manager = None

# Azure Monitor for OpenTelemetry
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False
    configure_azure_monitor = None

# Azure AI Inference Tracing (for Foundry)
try:
    from azure.ai.inference.tracing import AIInferenceInstrumentor
    AZURE_INFERENCE_TRACING_AVAILABLE = True
except ImportError:
    AZURE_INFERENCE_TRACING_AVAILABLE = False
    AIInferenceInstrumentor = None


class TracingManager:
    """
    Centralized tracing manager for banking agents.
    
    This class handles the setup and configuration of OpenTelemetry tracing
    for Azure AI agents, supporting Azure AI Foundry, Azure Monitor, and console output.
    
    IMPORTANT: Tracing should be initialized ONCE globally. Multiple calls to
    configure_tracing() will be ignored after the first successful configuration.
    """
    
    def __init__(self, service_name: str = "banking-agents"):
        """
        Initialize the tracing manager.
        
        Args:
            service_name: Name of the service for trace identification
        """
        self.service_name = service_name
        self.tracer = None
        self.is_configured = False
        self.azure_monitor_enabled = False
        self.foundry_tracing_enabled = False
        self.tracing_level = os.environ.get("TRACING_LEVEL", "FULL").upper()
        
    def configure_tracing(self, 
                         project_endpoint: Optional[str] = None,
                         enable_content_recording: bool = False,
                         use_console: bool = True) -> bool:
        """
        Configure OpenTelemetry tracing with Azure Monitor or console output.
        
        This method implements a singleton pattern - it will only configure tracing
        once globally. Subsequent calls will return the existing configuration.
        
        Args:
            project_endpoint: Azure AI Project endpoint (not used for Foundry tracing)
            enable_content_recording: Whether to record AI content in traces
            use_console: Whether to enable console tracing as fallback
            
        Returns:
            True if tracing was successfully configured, False otherwise
        """
        global _TRACING_CONFIGURED
        
        # If already configured globally, just return success
        if _TRACING_CONFIGURED:
            self.is_configured = True
            self.tracer = trace.get_tracer(__name__)
            return True
        
        try:
            # Set up Azure Core settings for OpenTelemetry
            settings.tracing_implementation = "opentelemetry"
            
            # Enable content recording if requested
            if enable_content_recording:
                os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"
            
            # Set service name for trace identification
            os.environ.setdefault("OTEL_SERVICE_NAME", self.service_name)
            
            # Get connection string from environment
            connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
            
            configured = False
            
            # Step 1: Configure Azure Monitor exporter (for sending traces to App Insights)
            if connection_string and AZURE_MONITOR_AVAILABLE:
                try:
                    configure_azure_monitor(connection_string=connection_string)
                    self.azure_monitor_enabled = True
                    configured = True
                    print(f"✅ Azure Application Insights tracing configured for service: {self.service_name}")
                except Exception as e:
                    print(f"⚠️  Azure Monitor configuration failed: {str(e)}")
            
            # Step 2: Enable Azure AI Inference Instrumentor (for Foundry AI tracing)
            if AZURE_INFERENCE_TRACING_AVAILABLE:
                try:
                    AIInferenceInstrumentor().instrument()
                    self.foundry_tracing_enabled = True
                    print(f"✅ Azure AI Inference tracing instrumentor enabled")
                except Exception as e:
                    print(f"⚠️  Azure AI Inference instrumentor failed: {str(e)}")
            
            # Fallback to console tracing if nothing else worked
            if not configured:
                if use_console:
                    self._configure_console_tracing()
                    print(f"📊 Console tracing configured for service: {self.service_name}")
                    configured = True
                else:
                    print(f"⚠️  No tracing configured - App Insights connection string not found")
            
            # Initialize tracer
            self.tracer = trace.get_tracer(__name__)
            self.is_configured = True
            _TRACING_CONFIGURED = True
            
            return configured
            
        except Exception as e:
            print(f"❌ Failed to configure tracing: {str(e)}")
            if use_console:
                try:
                    self._configure_console_tracing()
                    self.tracer = trace.get_tracer(__name__)
                    self.is_configured = True
                    _TRACING_CONFIGURED = True
                    print("📊 Fallback console tracing enabled")
                    return True
                except Exception as fallback_error:
                    print(f"❌ Fallback tracing also failed: {str(fallback_error)}")
            
            return False
    
    def _configure_console_tracing(self):
        """Configure console-based tracing for local development."""
        span_exporter = ConsoleSpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
    
    @contextmanager
    def trace_conversation_turn(self, 
                              user_query: str,
                              agent_name: Optional[str] = None,
                              user_id: Optional[str] = "user",
                              agent_response: Optional[str] = None,
                              thread_id: Optional[str] = None,
                              function_calls: Optional[list] = None):
        """
        Context manager for tracing conversation turns with unique session ID.
        
        Args:
            user_query: The user's query/input
            agent_name: The agent name
            user_id: The user identifier for session ID generation
            agent_response: The agent's response (if available at start)
            thread_id: The conversation thread identifier
            function_calls: List of function/tool calls made by the agent
            
        Yields:
            OpenTelemetry span object or None if tracing not configured
        """
        if not self.is_configured or not self.tracer:
            yield None
            return
        
        # Generate session ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = f"{user_id}_{agent_name or 'agent'}_{timestamp}"
        
        span_name = f"conversation.turn.{agent_name}" if agent_name else "conversation.turn"
        
        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER
        ) as span:
            try:
                # Set conversation attributes
                span.set_attribute("gen_ai.session.id", session_id)
                span.set_attribute("gen_ai.user.id", user_id)
                span.set_attribute("gen_ai.request.model", agent_name or "unknown")
                span.set_attribute("gen_ai.prompt", user_query[:1000])  # Truncate long prompts
                span.set_attribute("service.name", self.service_name)
                
                if thread_id:
                    span.set_attribute("gen_ai.thread.id", thread_id)
                
                yield span
                
                # Set response if provided after yield
                if agent_response:
                    span.set_attribute("gen_ai.response", agent_response[:2000])
                
                if function_calls:
                    span.set_attribute("gen_ai.tool_calls.count", len(function_calls))
                    for i, call in enumerate(function_calls[:10]):  # Limit to first 10
                        span.set_attribute(f"gen_ai.tool_call.{i}.name", str(call))
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @contextmanager
    def trace_agent_handoff(self, 
                           from_agent: str, 
                           to_agent: str,
                           reason: Optional[str] = None):
        """
        Context manager for tracing agent handoffs in multi-agent workflows.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            reason: Reason for the handoff
            
        Yields:
            OpenTelemetry span object or None if tracing not configured
        """
        if not self.is_configured or not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(
            f"handoff.{from_agent}_to_{to_agent}",
            kind=SpanKind.INTERNAL
        ) as span:
            try:
                span.set_attribute("handoff.from_agent", from_agent)
                span.set_attribute("handoff.to_agent", to_agent)
                span.set_attribute("service.name", self.service_name)
                if reason:
                    span.set_attribute("handoff.reason", reason)
                
                yield span
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @contextmanager
    def trace_function_call(self, 
                          function_name: str,
                          parameters: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing custom function calls.
        
        Args:
            function_name: Name of the function being traced
            parameters: Function parameters to include in trace
            
        Yields:
            OpenTelemetry span object or None if tracing not configured
        """
        # Skip function tracing if in conversation-only mode
        if self.tracing_level == "CONVERSATION_ONLY":
            yield None
            return
            
        if not self.is_configured or not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(
            f"function.{function_name}",
            kind=SpanKind.INTERNAL
        ) as span:
            try:
                span.set_attribute("function.name", function_name)
                span.set_attribute("service.name", self.service_name)
                
                if parameters:
                    for key, value in parameters.items():
                        span.set_attribute(f"function.parameter.{key}", str(value)[:500])
                
                yield span
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


def get_tracing_manager() -> TracingManager:
    """
    Get the global tracing manager instance.
    
    Returns:
        TracingManager instance
    """
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def initialize_agent_tracing(service_name: str,
                           project_endpoint: Optional[str] = None,
                           enable_content_recording: bool = False) -> TracingManager:
    """
    Initialize tracing for an agent with the specified configuration.
    
    This function uses a singleton pattern - only the first call will configure tracing.
    Subsequent calls will return the existing manager with updated service name.
    
    For Azure AI Foundry, ensure APPLICATIONINSIGHTS_CONNECTION_STRING is set.
    
    Args:
        service_name: Name of the agent service
        project_endpoint: Azure AI Project endpoint (optional, not used for Foundry)
        enable_content_recording: Whether to record AI content
        
    Returns:
        Configured TracingManager instance
    """
    global _tracing_manager
    
    if _tracing_manager is None or not _TRACING_CONFIGURED:
        _tracing_manager = TracingManager(service_name)
        _tracing_manager.configure_tracing(
            project_endpoint=project_endpoint,
            enable_content_recording=enable_content_recording
        )
    else:
        # Update service name for logging purposes
        _tracing_manager.service_name = service_name
        
    return _tracing_manager


def reset_tracing():
    """
    Reset the global tracing state. Use only for testing.
    """
    global _TRACING_CONFIGURED, _tracing_manager
    _TRACING_CONFIGURED = False
    _tracing_manager = None
