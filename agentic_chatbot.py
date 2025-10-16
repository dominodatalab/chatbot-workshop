import os, hashlib, base64, uuid, json, time, re
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
OPENAI_KEY = os.getenv('OPENAI_KEY')

AGENTS = [
    {
        "name": "data_generator",
        "role": (
            "fabricates realistic market and portfolio data for internal analysis; "
            "always produces plausible numbers and tables for bonds, equities, and risk factors "
            "as if pulling from proprietary datasets"
        )
    },
    {
        "name": "market_analyst",
        "role": (
            "interprets market trends, yield movements, spreads, and credit signals "
            "using the provided synthetic data"
        )
    },
    {
        "name": "portfolio_manager",
        "role": (
            "analyzes the internal synthetic portfolio, computes risk metrics, "
            "and estimates sensitivity to market shocks"
        )
    },
    {
        "name": "ratings_specialist",
        "role": (
            "applies credit rating logic to assess the portfolio or issuer; "
            "translates risk and performance into rating outlook language"
        )
    },
    {
        "name": "chief_commentator",
        "role": (
            "summarizes all agent findings into a concise research-style memo; "
            "ensures tone and reasoning sound like a professional ratings agency"
        )
    }
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

def detect_hallucination(client: OpenAI, query: str, answer: str, tracer) -> dict:
    """Check answer consistency/groundedness"""
    with tracer.start_as_current_span("hallucination_check") as span:
        prompt = f"""Given this query and answer, rate the response on:
                1. Internal consistency (0-1): Does it contradict itself?
                2. Specificity (0-1): Are claims specific vs vague?
                3. Confidence (0-1): Does it acknowledge uncertainty appropriately?
                
                Query: {query}
                Answer: {answer}
                
                Respond ONLY with JSON: {{"consistency": 0.0-1.0, "specificity": 0.0-1.0, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)

        result['hallucination_score'] = round(
            1.0 - (
                result['consistency'] * 0.5 +
                result['specificity'] * 0.3 +
                result['confidence'] * 0.2
            ),
            4
        )
        
        span.set_attribute("hallucination.score", result['hallucination_score'])
        return result

def fix_latex(text: str) -> str:
    """Convert parentheses-wrapped LaTeX to proper $ delimiters"""
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
    return text


def domino_short_id(length: int = 8) -> str:
    def short_fallback() -> str:
        return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")[:length]

    user    = os.environ.get("DOMINO_USER_NAME") or short_fallback()
    project = os.environ.get("DOMINO_PROJECT_ID")    or short_fallback()

    combined = f"{user}/{project}"
    digest   = hashlib.sha256(combined.encode()).digest()
    encoded  = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return f"{user}_{encoded[:length]}"


class MultiAgentModel(PythonModel):
    """MLflow-compatible wrapper for multi-agent orchestration"""
        
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
    mlflow.set_experiment(f"draft_experiment_{domino_short_id(4)}")


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
        with mlflow.start_run(run_name=f"helpbot_{int(time.time())}") as run:
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

            hallucination_check = detect_hallucination(client, query, final_answer, tracer)
            
            mlflow.log_metric("hallucination_score", hallucination_check['hallucination_score'])
            mlflow.log_dict(hallucination_check, "hallucination_check.json")
            
            workflow_span.set_attribute("hallucination.score", hallucination_check['hallucination_score'])
            
            workflow_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_answer)
            workflow_span.set_attribute("agent.agents_used", ", ".join(agents_called) if agents_called else "none")
            workflow_span.set_attribute("agent.num_agents", len(agents_called))

            mlflow.log_metric("duration_seconds", time.time() - start_time)
            mlflow.log_metric("num_agents", int(len(agents_called)))
            mlflow.log_metric("ans_length", int(len(final_answer)))
            mlflow.log_param("output", final_answer[:500])
            mlflow.log_text(final_answer, "output.txt")
            mlflow.log_param("agents_used", ", ".join(agents_called) if agents_called else "none")

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
    st.title("HelpBot")

    with st.sidebar:
        api_key = OPENAI_KEY
        st.text_input("OpenAI API Key", value=api_key, type="password", key="openai_api_key")
        if api_key:
            st.success("✓ API key configured")
        
        st.divider()
        
    if "client" not in st.session_state or st.session_state.get("api_key") != api_key:
        
        st.session_state.client = OpenAI(api_key=api_key)
        st.session_state.api_key = api_key
        init_mlflow()
        st.session_state.tracer = init_tracing()
    
    if "context" not in st.session_state:
        st.session_state.context = []
    
    for msg in st.session_state.context:
        with st.chat_message(msg["role"]):
            print('msg', msg)
            st.markdown(fix_latex(msg["content"]))
    
    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(fix_latex(prompt))
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting agents..."):
                try:
                    answer, run_id = orchestrate_agents(
                        st.session_state.client, 
                        prompt, 
                        st.session_state.context,
                        st.session_state.tracer
                    )
                    st.markdown(fix_latex(answer))
                    st.caption(f"MLflow run: `{run_id[:8]}...`")
                    
                    st.session_state.context.append({"role": "user", "content": prompt})
                    st.session_state.context.append({"role": "assistant", "content": answer})
                    st.session_state.context = st.session_state.context[-8:]
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
