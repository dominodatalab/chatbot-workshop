import os
import json
import time
from openai import OpenAI
import streamlit as st
import mlflow
from mlflow.pyfunc import PythonModel
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


class MultiAgentModel(PythonModel):
    """MLflow-compatible wrapper for multi-agent orchestration"""
    
    def load_context(self, context):
        """Initialize OpenAI client when model loads"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def _call_agent(self, agent_name: str, query: str) -> str:
        """Execute single agent query"""
        agent = next(a for a in AGENTS if a['name'] == agent_name)
        
        prompt = (
            f"You are a {agent['role']}. "
            f"Give a brief response (1-2 sentences).\n\n"
            f"Query: {query}"
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    
    def predict(self, context, model_input):
        """
        Predict method for MLflow model.
        Expects model_input as dict with 'query' and optional 'context' keys.
        """
        if isinstance(model_input, dict):
            query = model_input.get('query', '')
            conversation_context = model_input.get('context', [])
        else:
            query = model_input.iloc[0]['query']
            conversation_context = model_input.iloc[0].get('context', [])
        
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
        if conversation_context:
            messages.extend(conversation_context[-8:])
        messages.append({"role": "user", "content": query})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=AGENT_TOOLS,
            temperature=0.7
        )
        
        agents_called = []
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                agent_name = func_name.replace("call_", "")
                
                agent_response = self._call_agent(agent_name, args['query'])
                agents_called.append(agent_name)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": agent_response
                })
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.7
            )
        
        return {
            "answer": response.choices[0].message.content,
            "agents_used": agents_called,
            "num_agents": len(agents_called)
        }


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
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.7
            )
            
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
            
            workflow_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_answer)
            workflow_span.set_attribute("agent.agents_used", ",".join(agents_called))
            workflow_span.set_attribute("agent.num_agents", len(agents_called))
            
            mlflow.log_metric("duration_seconds", time.time() - start_time)
            mlflow.log_metric("num_agents", len(agents_called))
            mlflow.log_metric("ans_length", len(final_answer))
            mlflow.log_metric("output", final_answer))
            mlflow.log_text(final_answer, "output.txt")
            if agents_called:
                mlflow.log_param("agents_used", ",".join(agents_called))
            
            # Save agent configuration as JSON
            agent_config = {
                "agents": AGENTS,
                "agent_tools": AGENT_TOOLS,
                "orchestrator_prompt": (
                    "You're an orchestrator. Use available agent tools to gather "
                    "perspectives, then synthesize a final answer (2-3 sentences max). "
                    "Be conversational and direct."
                ),
                "agent_prompt_template": (
                    "You are a {role}. "
                    "Give a brief response (1-2 sentences).\n\n"
                    "Query: {query}"
                ),
                "model_params": {
                    "orchestrator_model": "gpt-4o-mini",
                    "orchestrator_temperature": 0.7,
                    "agent_model": "gpt-4o-mini",
                    "agent_temperature": 0.8,
                    "agent_max_tokens": 150
                }
            }
            mlflow.log_dict(agent_config, "agent_config.json")
            
            # Log model as generic Python model
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MultiAgentModel(),
                pip_requirements=[
                    "openai",
                    "arize-otel",
                    "openinference-instrumentation-openai",
                    "opentelemetry-api"
                ]
            )
            
            return final_answer, run.info.run_id


def main():
    st.title("Multi-Agent Assistant")
    
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
    
    if not api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar to begin")
        return
    
    if "client" not in st.session_state or st.session_state.get("api_key") != api_key:
        st.session_state.client = OpenAI(api_key=api_key)
        st.session_state.api_key = api_key
        init_mlflow()
        st.session_state.tracer = init_tracing()
    
    if "context" not in st.session_state:
        st.session_state.context = []
    
    for msg in st.session_state.context:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
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
                    
                    st.session_state.context.append({"role": "user", "content": prompt})
                    st.session_state.context.append({"role": "assistant", "content": answer})
                    st.session_state.context = st.session_state.context[-8:]
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()