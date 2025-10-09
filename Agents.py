from openai import OpenAI
import streamlit as st
import asyncio
import mlflow
import time
import os
from typing import Dict, Any
from opentelemetry.trace import get_tracer

# Config (removed hardcoded API key)
ARIZE_API_KEY = 'ak-83932695-b8e5-4c1b-b06d-300dbd28ea1b-A2RReorb1KqILlL3sZ9KXh2YFrdDBZd8'
ARIZE_SPACE_ID = 'U3BhY2U6MjY4ODI6bmYyUg=='

def mask_api_key(api_key: str) -> str:
    """Mask API key showing only first 3 and last 3 characters"""
    if len(api_key) < 6:
        return "***"
    return f"{api_key[:3]}...{api_key[-3:]}"

# MLflow setup
@st.cache_resource
def init_mlflow():
    user = os.environ.get("DOMINO_STARTING_USERNAME", "demo_user")
    experiment_name = f"MultiAgent_Decisions_{user}"
    mlflow.set_experiment(experiment_name)
    return experiment_name

# Arize setup
@st.cache_resource
def init_arize():
    from arize.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    
    exporter = OTLPSpanExporter(
        endpoint="https://otlp.arize.com/v1/traces",
        headers={"space_id": ARIZE_SPACE_ID, "api_key": ARIZE_API_KEY}
    )
    
    tp = register(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY, project_name="FSI-Demo-Project")
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    
    OpenAIInstrumentor().instrument(tracer_provider=tp)
    return tp

# Initialize
experiment_name = init_mlflow()
tracer_provider = init_arize()

class Agent:
    def __init__(self, name: str, role: str, client: OpenAI):
        self.name = name
        self.role = role
        self.client = client
        self.tracer = get_tracer(__name__)

    async def respond(self, message: str, context: str = "") -> str:
        with self.tracer.start_as_current_span(f"{self.name.lower().replace(' ', '_')}_response") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("input.message", message[:200])
            
            system_prompt = f"You are {self.name}, {self.role}. {context}\nBe confident and specific."
            
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=1.0,
                max_tokens=400
            )
            
            response = resp.choices[0].message.content
            span.set_attribute("output.response", response[:200])
            return response

class MultiAgentOrchestrator:
    def __init__(self, client: OpenAI, referring_mode: bool = False):
        self.client = client
        self.referring_mode = referring_mode
        self.tracer = get_tracer(__name__)
        
        self.research_agent = Agent("Research Analyst", "an expert researcher", client)
        self.creative_agent = Agent("Creative Strategist", "an innovation expert", client)
        self.referee_agent = Agent("Referee", "a synthesist who adds concrete details", client)

    async def collaborate(self, user_query: str) -> Dict[str, Any]:
        session_id = f"session_{hash(user_query[:50])}"
        
        # Single MLflow run
        with mlflow.start_run(run_name=f"decision_{session_id[:8]}"):
            start_time = time.time()
            
            # Log basic params
            mlflow.log_param("query", user_query[:100])
            mlflow.log_param("referring_mode", self.referring_mode)
            
            with self.tracer.start_as_current_span("multi_agent_collaboration") as main_span:
                main_span.set_attribute("session.id", session_id)
                main_span.set_attribute("user.query", user_query)
                
                # Research
                research = await self.research_agent.respond(f"Analyze: {user_query}")
                
                # Creative
                creative = await self.creative_agent.respond(
                    f"Generate solutions for: {user_query}", 
                    f"Research context: {research}"
                )
                
                # Referee (if enabled)
                referee = ""
                if self.referring_mode:
                    referee = await self.referee_agent.respond(
                        f"Add specific details for: {user_query}",
                        f"Research: {research}\nCreative: {creative}"
                    )
                
                # Final synthesis
                synth_context = f"Research: {research}\nCreative: {creative}\nDetails: {referee}"
                synthesis = await self.research_agent.respond(
                    f"Give final decisive answer for: {user_query}",
                    synth_context
                )
                
                main_span.set_attribute("workflow.status", "completed")
            
            # Log metrics
            duration = time.time() - start_time
            mlflow.log_metric("duration_seconds", duration)
            mlflow.log_metric("agents_used", 3 if self.referring_mode else 2)
            
            # Log outputs
            mlflow.log_text(research, "research.txt")
            mlflow.log_text(creative, "creative.txt")
            if referee:
                mlflow.log_text(referee, "referee.txt")
            mlflow.log_text(synthesis, "synthesis.txt")
            
            return {
                "research": research,
                "creative": creative,
                "referee": referee,
                "synthesis": synthesis,
                "session_id": session_id,
                "run_id": mlflow.active_run().info.run_id
            }

# UI
st.title("Multi-Agent Collaboration Demo")

with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key",
        help="Your API key is stored securely in memory only"
    )
    
    # Show masked key if provided
    if api_key:
        st.success(f"API Key set: {mask_api_key(api_key)}")
    else:
        st.warning("Please enter your OpenAI API key to continue")
    
    st.divider()
    
    referring_mode = st.toggle("Add Creative Details", value=True)
    st.info(f"Experiment: {experiment_name}")

# Initialize client only if API key is provided
client = None
if api_key:
    if "client" not in st.session_state or st.session_state.get("api_key") != api_key:
        st.session_state.client = OpenAI(api_key=api_key)
        st.session_state.api_key = api_key
    client = st.session_state.client

if "conversations" not in st.session_state:
    st.session_state.conversations = []

# Show history only if client is available
if client:
    # History
    for i, convo in enumerate(st.session_state.conversations):
        with st.expander(f"{convo['query'][:50]}... (Run: {convo['run_id'][:8]})"):
            st.write("**Research:**", convo['research'])
            st.write("**Creative:**", convo['creative'])
            if convo['referee']:
                st.write("**Details:**", convo['referee'])
            st.write("**Final Answer:**", convo['synthesis'])

    # Chat
    if prompt := st.chat_input("Ask something"):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Working..."):
                orchestrator = MultiAgentOrchestrator(client, referring_mode)
                result = asyncio.run(orchestrator.collaborate(prompt))
                
                st.write("**Research Analysis:**")
                st.write(result["research"])
                
                st.write("**Creative Ideas:**") 
                st.write(result["creative"])
                
                if result["referee"]:
                    st.write("**Additional Details:**")
                    st.write(result["referee"])
                    
                st.write("**Final Answer:**")
                st.write(result["synthesis"])
                
                st.caption(f"MLflow Run: {result['run_id']}")
                
                st.session_state.conversations.append({
                    "query": prompt,
                    "research": result["research"],
                    "creative": result["creative"], 
                    "referee": result["referee"],
                    "synthesis": result["synthesis"],
                    "run_id": result["run_id"]
                })
else:
    st.info("👈 Please enter your OpenAI API key in the sidebar to start chatting")