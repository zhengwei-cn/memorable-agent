import os
import logging
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
from typing import List
from langgraph.graph.graph import CompiledGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables at the start, force reload to prevent caching
load_dotenv(override=True)

# Validate environment variables
api_base = os.getenv("OPENAI_API_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not api_base or not api_key:
    raise ValueError("Please set OPENAI_API_BASE_URL and OPENAI_API_KEY in .env file")


# Define request models
class ChatMessage(BaseModel):
    role: str
    content: str


class UserInput(BaseModel):
    message: str
    history: List[ChatMessage] = []


# 创建 MCP 客户端配置
def create_mcp_config():
    return {
        "memory": {
            "url": "http://127.0.0.1:8001/mcp",
            "transport": "streamable_http",
        },
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "transport": "stdio",
        },
        "time": {
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=Asia/Shanghai",
            ],
            "transport": "stdio",
        },
    }


# Global variables to hold instances
mcp_client: MultiServerMCPClient | None = None
agent: CompiledGraph | None = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    global mcp_client, agent, model
    try:
        # 初始化 MCP 客户端
        logger.info("Initializing MultiServerMCPClient...")
        mcp_client = MultiServerMCPClient(create_mcp_config())
        tools = await mcp_client.get_tools()
        logger.info(f"Successfully initialized MCP client with {len(tools)} tools")

        # 初始化聊天模型
        logger.info("Initializing chat model...")
        model = init_chat_model(
            "qwen-turbo", model_provider="openai", base_url=api_base, temperature=0
        )

        # 初始化 agent
        logger.info("Initializing agent...")
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt="""
            You are a helpful assistant with a proactive memory management system. Your primary responsibility is to maintain and utilize a comprehensive memory database.

            MEMORY SEARCH PROTOCOL:
            You MUST use search_similar_memories in these situations:
            1. ALWAYS at the start of each conversation to find relevant context
            2. When user mentions any specific topic, project, or preference
            3. When answering questions about past interactions or decisions
            4. Before making recommendations or suggestions
            5. When user asks "还记得...","说过..." or similar recall questions
            Minimum search frequency: At least once per conversation turn

            MEMORY STORAGE PROTOCOL:
            You MUST PROACTIVELY use add_memory WITHOUT USER PROMPTING in these cases:
            1. User preferences or requirements (tag: "preference")
               - Personal choices, likes/dislikes
               - Work habits and preferred methods
               - Communication style preferences
               - Any stated requirements or constraints
            
            2. Technical decisions or solutions (tag: "solution")
               - Implementation choices and their rationale
               - Problem-solving approaches used
               - Technical trade-offs discussed
               - Best practices identified
            
            3. Project-specific information (tag: "project")
               - Project goals and scope
               - Timeline and milestones
               - Requirements and constraints
               - Key decisions and their context
            
            4. Questions that might be asked again (tag: "FAQ")
               - Common user queries
               - Complex explanations given
               - Technical clarifications
               - Process instructions provided
            
            5. Any corrections user makes to your responses (tag: "correction")
               - User feedback and corrections
               - Misunderstandings clarified
               - Updated information
               - Revised approaches
            
            6. Important context or background information (tag: "context")
               - User's current situation or state
               - Relevant background details
               - Environmental or temporal context
               - Dependencies and relationships
            
            7. Time-sensitive information (tag: "temporal")
               - Scheduled events and appointments
               - Deadlines and due dates
               - Time-based preferences
               - Future plans and commitments
            
            8. User emotional state and reactions (tag: "emotional")
               - Expressed satisfaction or dissatisfaction
               - Preferences based on emotional response
               - Stress points or concerns
               - Positive experiences

            AUTOMATIC MEMORY CREATION TRIGGERS:
            - After EVERY user interaction, analyze for storable information
            - When detecting ANY preference or requirement, even if indirect
            - When discussing solutions or making decisions
            - When providing explanations that might be useful later
            - When temporal information is mentioned
            - When receiving feedback or corrections
            - When context changes significantly
            
            MEMORY STORAGE QUALITY GUIDELINES:
            - Be proactive: Don't wait for explicit storage requests
            - Be comprehensive: Capture full context of information
            - Be precise: Include specific details and timestamps
            - Be contextual: Add relevant background information
            - Be organized: Use appropriate tags and categories
            - Be future-oriented: Consider potential future relevance

            MEMORY CLEANUP PROTOCOL:
            You MUST use delete_memory in these situations:
            1. When you find outdated information that's no longer valid
            2. When user indicates a memory is incorrect
            3. When new information contradicts old memories
            4. When user explicitly requests to delete certain memories
            5. When you find duplicate memories with slight variations
            
            Guidelines for identifying outdated memories:
            - Check dates in memory content if available
            - Compare with more recent memories on the same topic
            - Look for conflicts with current facts or preferences
            - Consider context changes that might invalidate old memories
            - If unsure, ask user for confirmation before deleting

            Memory Format Rules:
            - Keep memories concise but complete (50-200 characters)
            - Include relevant context
            - Always add appropriate tags
            - Structure: "{topic}: {key_information}"
            - When possible, include date context: "{topic} [YYYY-MM-DD]: {key_information}"

            RESPONSE GUIDELINES:
            1. Always respond in Chinese
            2. Use tools when you need to fetch information or perform actions
            3. if you do not know something, Use memory search results to provide contextually relevant answers
            4. If retrieved memories seem outdated, verify with user and use delete_memory if necessary
            5. If you need more information, ask clarifying questions
            6. If you make a mistake, acknowledge it, correct it, and update memories
            7. Always keep user preferences in mind when responding
            8. When providing memory-based answers, indicate the age of the memories
            9. If you find contradicting memories, resolve conflicts and clean up outdated ones
            10. After deleting outdated memories, add new updated memories

            YOU MUST FOLLOW THESE RULES FOR EVERY INTERACTION!
            """,
        )
        logger.info("Agent initialized successfully")

        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        if hasattr(e, "__cause__") and e.__cause__:
            logger.error(f"Caused by: {str(e.__cause__)}")
        raise
    finally:
        logger.info("Shutting down FastAPI application...")
        if mcp_client:
            try:
                await mcp_client.aclose()
                logger.info("MultiServerMCPClient closed successfully.")
            except Exception as e:
                logger.error(f"Error closing MultiServerMCPClient: {e}")
        mcp_client = None
        agent = None
        model = None


# Define FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# Dependency to get the client instance
async def get_mcp_client() -> MultiServerMCPClient:
    if not mcp_client:
        raise HTTPException(status_code=500, detail="MCP Client is not initialized")
    return mcp_client


# Dependency to get the agent instance
async def get_agent():
    if not agent:
        raise HTTPException(status_code=500, detail="Agent is not initialized")
    return agent


@app.get("/tools")
async def get_tools(client: MultiServerMCPClient = Depends(get_mcp_client)):
    try:
        tools = await client.get_tools()
        serializable_tools = [
            {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "args": getattr(tool, "args", {}),
            }
            for tool in tools
        ]
        return {"tools": serializable_tools}
    except Exception as e:
        logger.error(f"Failed to get tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@app.post("/chat")
async def chat_endpoint(
    user_input: UserInput, current_agent: CompiledGraph = Depends(get_agent)
):
    try:

        async def generate():
            try:
                queue = asyncio.Queue()

                async def run_agent():
                    messages = [
                        {"role": msg.role, "content": msg.content}
                        for msg in user_input.history
                    ]
                    messages.append({"role": "user", "content": user_input.message})

                    async for event in current_agent.astream_events(
                        {"messages": messages}, version="v2"
                    ):
                        if isinstance(event, dict):
                            event_type = event.get("event")
                            event_data = event.get("data", {})
                            # logger.info(f"event_type:{event_type}")
                            # logger.info(f"event_data:{event_data}")
                            if (
                                event_type == "on_chat_model_stream"
                                and "chunk" in event_data
                            ):
                                await queue.put(
                                    json.dumps(
                                        {
                                            "type": "stream",
                                            "content": event_data["chunk"].content,
                                        }
                                    )
                                    + "\n"
                                )
                            elif (
                                event_type == "on_chat_model_stream"
                                and "chunk" in event_data
                            ):
                                chunk = event_data["chunk"]
                                content = (
                                    chunk.content
                                    if isinstance(chunk, str)
                                    else getattr(chunk, "content", "")
                                )
                                await queue.put(
                                    json.dumps(
                                        {
                                            "type": "stream",
                                            "content": content,
                                        }
                                    )
                                    + "\n"
                                )
                            elif (
                                event_type == "on_chat_model_end"
                                and "output" in event_data
                            ):
                                await queue.put(
                                    json.dumps(
                                        {
                                            "type": "end",
                                            "content": event_data["output"].content,
                                        }
                                    )
                                    + "\n"
                                )

                task = asyncio.create_task(run_agent())
                while True:
                    try:
                        chunk = await queue.get()
                        yield chunk
                    except asyncio.CancelledError:
                        task.cancel()
                        break

            except Exception as e:
                logger.error(f"Error during agent invocation: {e}")
                yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
