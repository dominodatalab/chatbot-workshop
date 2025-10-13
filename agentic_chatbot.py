import os
import json
import time
from openai import OpenAI
import streamlit as st
import mlflow
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues


# Configuration
ARIZE_API_KEY = os.getenv('ARIZE_API_KEY')
ARIZE_SPACE_ID = os.getenv('ARIZE_SPACE_ID')

AGENTS = [
    {"name": "researcher", "role": "expert at finding facts and data"},
    {"name": "creative", "role": "brainstormer with wild ideas"},
    {"name": "critic", "role": "skeptic who finds flaws"}
]

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"call_{agent['name']}",
            "description": f"Call the {agent['name']} agent ({agent['role']})",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": f"Question for the {agent['name']}"}
                },
                "required": ["query"]
            }
        }
    }
    for agent in AGENTS
]


@st.cache_resource
def init_mlflow():
    """Initialize MLflow experiment"""
    user = os.getenv("DOMINO_STARTING_USERNAME", "demo_user")
    mlflow.set_experiment(f"Agent_POC_{user}")


@st.cache_resource
def init_tracing():
    """Initialize Arize tracing with HTTP (faster for dev/debug)"""
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    
    tracer_provider = register(
        space_id=ARIZE_SPACE_ID,
        api_key=ARIZE_API_KEY,
        project_name="Agent-POC",
    )
    
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # Return tracer for manual workflow spans
    return trace.get_tracer(__name__)


def call_agent(client: OpenAI, agent_name: str, query: str, tracer) -> str:
    """Execute single agent query with tracing"""
    agent = next(a for a in AGENTS if a['name'] == agent_name)
    
    with tracer.start_as_current_span(
        f"agent.{agent_name}",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.INPUT_VALUE: query,
            "agent.name": agent_name,
            "agent.role": agent['role'],
        }
    ) as span:
        prompt = (
            f"You are a {agent['role']}. "
            f"Give a brief response (1-2 sentences).\n\n"
            f"Query: {query}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=150
        )
        
        result = response.choices[0].message.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result)
        
        return result


def orchestrate_agents(client: OpenAI, query: str, context: list, tracer) -> tuple[str, str]:
    """
    Orchestrate multi-agent workflow with MLflow tracking and OTEL tracing.
    Returns (final_answer, run_id)
    """
    with tracer.start_as_current_span(
        "agent_workflow",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            SpanAttributes.INPUT_VALUE: query,
        }
    ) as workflow_span:
        with mlflow.start_run(run_name=f"agents_{int(time.time())}") as run:
            start_time = time.time()
            mlflow.log_param("query", query[:100])
            
            # Build conversation history
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You're an orchestrator. Use available agent tools to gather "
                        "perspectives, then synthesize a final answer (2-3 sentences max). "
                        "Be conversational and direct."
                    )
                }
            ]
            messages.extend(context[-8:])
            messages.append({"role": "user", "content": query})
            
            # Initial orchestrator call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.7
            )
            
            # Handle agent tool calls
            agents_called = []
            while response.choices[0].message.tool_calls:
                messages.append(response.choices[0].message)
                
                for tool_call in response.choices[0].message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    agent_name = func_name.replace("call_", "")
                    
                    agent_response = call_agent(client, agent_name, args['query'], tracer)
                    agents_called.append(agent_name)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": agent_response
                    })
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=AGENT_TOOLS,
                    temperature=0.7
                )
            
            final_answer = response.choices[0].message.content
            
            # Set workflow span attributes
            workflow_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_answer)
            workflow_span.set_attribute("agent.agents_used", ",".join(agents_called))
            workflow_span.set_attribute("agent.num_agents", len(agents_called))
            
            # Log metrics
            mlflow.log_metric("duration_seconds", time.time() - start_time)
            mlflow.log_metric("num_agents_called", len(agents_called))
            mlflow.log_text(final_answer, "output.txt")
            if agents_called:
                mlflow.log_param("agents_used", ",".join(agents_called))
            
            return final_answer, run.info.run_id


def main():
    st.title("Multi-Agent Assistant")
    
    # Sidebar configuration
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        if api_key:
            st.success("✓ API key configured")
        
        st.divider()
        st.caption(
            "**Available Agents:**\n"
            "• Researcher (facts & data)\n"
            "• Creative (brainstorming)\n"
            "• Critic (evaluation)\n\n"
        )
    
    # Early return if no API key
    if not api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar to begin")
        return
    
    # Initialize resources
    if "client" not in st.session_state or st.session_state.get("api_key") != api_key:
        st.session_state.client = OpenAI(api_key=api_key)
        st.session_state.api_key = api_key
        init_mlflow()
        st.session_state.tracer = init_tracing()
    
    if "context" not in st.session_state:
        st.session_state.context = []
    
    # Display conversation history
    for msg in st.session_state.context:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting agents..."):
                try:
                    answer, run_id = orchestrate_agents(
                        st.session_state.client, 
                        prompt, 
                        st.session_state.context,
                        st.session_state.tracer
                    )
                    st.write(answer)
                    st.caption(f"📊 MLflow run: `{run_id[:8]}...`")
                    
                    # Update context (keep last 8 messages)
                    st.session_state.context.append({"role": "user", "content": prompt})
                    st.session_state.context.append({"role": "assistant", "content": answer})
                    st.session_state.context = st.session_state.context[-8:]
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()