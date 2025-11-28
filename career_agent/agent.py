"""Factory for the LangChain agent used inside Streamlit."""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

from .config import DEFAULT_SYSTEM_PROMPT
from .data import CandidateContext
from .tools import build_tools


def build_agent(
    context: CandidateContext,
    model_name: str,
    temperature: float,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentExecutor:
    """Create an AgentExecutor configured for the given candidate context."""
    tools = build_tools(context)
    system_message = (
        f"{system_prompt}\n\n"
        f"Context:\n"
        f"- Target role: {context.target_role}\n"
        f"- Location: {context.location}\n"
        f"- Experience: {context.years_experience:.1f} years\n"
        f"- Skills: {', '.join(context.skills) or 'No skills provided'}\n"
        f"- Resume snippet: {context.resume_text[:600]}...\n\n"
        "You have access to the following tools:\n"
        "{tools}\n"
        "Only invoke tools whose names appear in this list. Tool names: {tool_names}.\n\n"
        "When reasoning, you MUST strictly follow this protocol and start every step with `Thought:`. "
        "Do not address the user until you produce the final response—no greetings, apologies, or explanations mid-chain.\n"
        "Thought: describe what you are considering or whether a tool is needed. This line is mandatory before any action. "
        "Keep the thought concise and reference the exact user instruction you are satisfying.\n"
        "Action: the exact tool name you wish to call with NO punctuation or code fences (e.g., `Action: salary_estimator`). This line MUST appear immediately after the Thought when a tool is required.\n"
        "Action Input: a STRICTLY valid JSON object that matches the tool schema (double quotes around keys/strings). "
        "Do not use equals signs, single quotes, or code fences. Example: `Action Input: {{\"job_title\": \"Data Scientist\"}}`.\n"
        "Observation: (the system will supply the tool result—wait for it before continuing). Do not add commentary in the same turn; resume with a new Thought or proceed to the Final Answer.\n"
        "The order MUST always be Thought → Action → Action Input → Observation. Never omit or reorder these lines.\n"
        "Example of correct formatting:\n"
        "Thought: Need the salary tool to answer the user's question.\n"
        "Action: salary_estimator\n"
        "Action Input: {{\"job_title\": \"Machine Learning Engineer\", \"location\": \"Seattle, WA\", \"years_experience\": 6}}\n"
        "Observation: <result provided by the system>\n"
        "Thought: Summarize the observation for the user.\n"
        "Final Answer: <response>\n"
        "Never call a tool unless the user request requires it. If the user explicitly asks to run a specific tool (e.g., \"score my resume\"), call only that tool unless additional context is required to answer. "
        "Always propagate any role/location/experience mentioned in the latest user message into the tool arguments.\n"
        " Do not call multiple tools just because they exist; each action must be justified by the instruction in your current Thought.\n"
        "Use at most **two** tool invocations per user request and never call the same tool twice in a row unless the user explicitly asks for more detail. "
        "If the user only greets you or asks something answerable without structured data, skip tools and move directly to the final answer.\n"
        "After you have gathered the necessary information (or decide no tool is required), synthesize everything and respond with:\n"
        "Final Answer: <your completed response>\n"
        "Once you output `Final Answer`, STOP immediately—no further thoughts or actions.\n"
        "Never output text outside of this schema."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    llm = ChatOllama(model=model_name, temperature=temperature)
    agent = create_react_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20,
        early_stopping_method="force",
        callbacks=[StdOutCallbackHandler()],
    )
    return executor

