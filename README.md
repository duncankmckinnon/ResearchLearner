# Research Learning Agent

An AI-powered research assistant that can find, analyze, and synthesize academic papers, build knowledge over time, and provide intelligent insights. This agent uses LangGraph for workflow orchestration, arXiv for paper discovery, and Mem0 for persistent knowledge management.

## Features

### 🔬 **Research Capabilities**
- **Paper Discovery**: Search arXiv for relevant academic papers by topic or keywords
- **Paper Analysis**: Download and analyze paper content, extracting key insights
- **Knowledge Synthesis**: Combine findings from multiple papers into comprehensive summaries
- **Citation Management**: Track and reference analyzed papers

### 🧠 **Knowledge Management**
- **Persistent Memory**: Build a knowledge graph of learned concepts using Mem0
- **Knowledge Querying**: Ask questions about previously learned topics
- **Cross-Session Learning**: Knowledge persists across conversations and sessions
- **Contextual Retrieval**: Find related information from the knowledge base

### 🤖 **Intelligent Workflow**
- **Intent Detection**: Automatically determines if you want research, knowledge queries, or analysis
- **Dynamic Planning**: Creates step-by-step execution plans based on your request
- **Real-time Progress**: Streaming interface shows progress through research workflow
- **Error Handling**: Graceful degradation and informative error messages

### 💬 **Interactive Demo UI**
- **Chat Interface**: Modern, responsive web interface for conversations
- **Markdown Support**: Rich text formatting for better readability
- **Progress Tracking**: Visual progress bars showing research workflow steps
- **Streaming Responses**: Real-time updates as the agent works

### 📊 **Observability**
- **Phoenix Integration**: Full tracing and monitoring of agent workflows
- **Request Tracking**: Monitor performance and debug issues
- **OpenTelemetry**: Comprehensive observability for all components

## Architecture

The agent uses a modular LangGraph-based architecture:

```
User Request → Intent Detection → Planning → Execution → Response Generation
                                              ↓
                                    ┌─ Research Execution
                                    │  ├─ Topic Extraction
                                    │  ├─ Paper Search (arXiv)
                                    │  ├─ Paper Analysis
                                    │  └─ Knowledge Storage
                                    │
                                    └─ Knowledge Query
                                       ├─ Knowledge Search
                                       ├─ Information Retrieval
                                       └─ Response Formulation
```

### Core Components

- **LangGraph Agent**: Orchestrates the research workflow with multiple specialized nodes
- **arXiv Client**: Searches and downloads academic papers from arXiv
- **Knowledge Graph**: Mem0-powered persistent memory for long-term learning
- **Prompts System**: Centralized prompt management for all AI interactions
- **Demo Interface**: Flask-powered web UI with streaming capabilities
- **FastAPI Server**: High-performance API server with OpenTelemetry tracing

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ with pyenv
- OpenAI API key
- Internet connection for arXiv access

## Quick Start

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd ResearchLearner
   ./bin/bootstrap.sh
   source .venv/bin/activate
   ```

2. **Configure Environment Variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY="your-openai-api-key"
   OPENAI_MODEL="gpt-4o"
   OPENAI_TEMPERATURE=0.1
   FASTAPI_URL="http://fastapi:8000"
   PHOENIX_COLLECTOR_ENDPOINT="http://phoenix:6006/v1/traces"
   ```

3. **Run the Application**
   ```bash
   ./bin/run_agent.sh --build
   ```

4. **Access the Interfaces**
   - **Demo Chat**: [http://localhost:8080](http://localhost:8080)
   - **Phoenix Dashboard**: [http://localhost:6006](http://localhost:6006)

## Usage Examples

### Research Mode
Ask the agent to research any academic topic:
- *"Find papers about transformer architectures"*
- *"Research quantum computing applications in machine learning"*
- *"What are the latest developments in JEPA models?"*

The agent will:
1. Extract research keywords
2. Search arXiv for relevant papers
3. Download and analyze top papers
4. Store findings in knowledge graph
5. Provide comprehensive synthesis

### Knowledge Query Mode
Query previously learned information:
- *"What have I learned about transformers?"*
- *"Summarize what you know about neural networks"*
- *"Tell me about the papers on reinforcement learning"*

### Analysis Mode
Analyze specific papers:
- *"Analyze the paper arxiv:2024.12345"*
- *"What are the key insights from the BERT paper?"*

## Configuration

### OpenAI Configuration
```env
OPENAI_API_KEY="your-api-key"
OPENAI_MODEL="gpt-4o"                    # or gpt-4o-mini, gpt-3.5-turbo
OPENAI_TEMPERATURE=0.1                   # 0.0-1.0, lower = more focused
```

### Service URLs (for Docker)
```env
FASTAPI_URL="http://fastapi:8000"
PHOENIX_COLLECTOR_ENDPOINT="http://phoenix:6006/v1/traces"
```

## Development

### Project Structure
```
agent/
├── agent.py              # Main agent orchestrator
├── langgraph_agent.py    # LangGraph workflow implementation
├── arxiv_client.py       # arXiv paper search and download
├── knowledge_graph.py    # Mem0 knowledge management
├── prompts.py            # Centralized prompt templates
├── schema.py             # Request/response models
├── server.py             # FastAPI application server
├── caching.py            # LRU cache for conversations
└── demo_code/            # Web demo interface
    ├── demo_server.py    # Flask demo server
    ├── templates/        # HTML templates
    └── static/           # CSS, JavaScript, assets
```

### Key Components

#### Agent Workflow (LangGraph)
The agent uses a sophisticated LangGraph workflow with multiple specialized nodes:

- **IntentDetectionNode**: Determines user intent (research, knowledge_query, analysis, general)
- **PlanningNode**: Creates step-by-step execution plans
- **ResearchExecutionNode**: Handles paper search, analysis, and synthesis
- **KnowledgeQueryNode**: Retrieves and formulates responses from knowledge base
- **ResponseGenerationNode**: Creates final user-facing responses

#### Research Capabilities
- **Paper Search**: Advanced arXiv queries with category filtering
- **Content Analysis**: Full paper download and AI-powered analysis
- **Knowledge Storage**: Automatic storage of insights in persistent memory
- **Cross-Reference**: Links related papers and concepts

#### Knowledge Management
- **Mem0 Integration**: Vector-based knowledge storage with semantic search
- **Persistent Learning**: Knowledge survives across sessions and conversations
- **Contextual Retrieval**: Smart retrieval of relevant information
- **Knowledge Synthesis**: Combines multiple sources for comprehensive answers

### Customization

#### Adding New Research Sources
To add research sources beyond arXiv, extend the `SimpleResearchAgent` class:

```python
class SimpleResearchAgent:
    async def search_papers(self, query: str, sources: List[str] = ["arxiv"]):
        if "pubmed" in sources:
            # Add PubMed search logic
        if "semantic_scholar" in sources:
            # Add Semantic Scholar search logic
```

#### Custom Prompts
All prompts are centralized in `agent/prompts.py`. Add new prompts as properties:

```python
@property
def custom_analysis_prompt(self) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "Your custom system prompt..."),
        ("human", "{user_input}")
    ])
```

#### Extending Knowledge Graph
The knowledge graph can be extended to store custom metadata:

```python
def add_custom_insight(self, insight: str, metadata: Dict):
    self.memory.add(
        messages=[{"role": "user", "content": insight}],
        user_id=self.user_id,
        metadata=metadata
    )
```

## Monitoring and Debugging

### Phoenix Dashboard
Access the Phoenix dashboard at [http://localhost:6006](http://localhost:6006) to:
- View request traces and spans
- Monitor agent workflow execution
- Debug performance issues
- Analyze usage patterns

### Logs
View container logs:
```bash
docker logs researchlearner-fastapi-1    # Agent logs
docker logs researchlearner-demo-1       # Demo server logs  
docker logs researchlearner-phoenix-1    # Phoenix logs
```

### Health Checks
Check service health:
```bash
curl http://localhost:8080/api/health     # Demo server
curl http://localhost:8000/health         # FastAPI server
```

## Troubleshooting

### Common Issues

**Phoenix Connection Error**
- Ensure Phoenix container is running: `docker ps`
- Check PHOENIX_COLLECTOR_ENDPOINT in .env file

**API Key Issues**
- Verify OPENAI_API_KEY is valid and has sufficient credits
- Check OPENAI_MODEL is supported (gpt-4o, gpt-4o-mini, etc.)

**arXiv Access Issues**
- Ensure internet connectivity
- arXiv may rate limit requests; the client handles this automatically

**Knowledge Graph Issues**
- Check that `~/.research_learner/knowledge_db` directory is writable
- Verify Mem0 installation: `pip install mem0ai`

**Container Build Issues**
```bash
./bin/run_agent.sh --build              # Rebuild containers
docker system prune                     # Clean up old containers
docker-compose logs                     # View detailed logs
```

**Demo UI Issues**
- Hard refresh browser (Ctrl+F5 / Cmd+Shift+R)
- Check browser console for JavaScript errors
- Verify streaming is supported in your browser

### Performance Optimization

**Research Performance**
- Adjust `max_papers` parameter in research queries
- Use specific categories for arXiv searches
- Cache expensive operations

**Knowledge Graph Performance**
- Regular cleanup of old memories
- Optimize vector store configuration
- Use semantic search filters

**API Performance**
- Monitor OpenAI API usage and costs
- Implement request batching for multiple papers
- Use appropriate model selection (gpt-4o-mini for simpler tasks)

## Dependencies

### Core Dependencies
- **LangGraph**: Agent workflow orchestration
- **OpenAI**: LLM provider for analysis and synthesis  
- **Mem0**: Knowledge graph and long-term memory
- **arXiv**: Academic paper search and download
- **FastAPI**: High-performance API server
- **Phoenix**: Observability and tracing

### Demo Dependencies
- **Flask**: Demo web server
- **Bootstrap**: UI styling
- **JavaScript**: Streaming chat interface

## Claude Desktop Integration (MCP)

This research agent can be connected to Claude Desktop as an MCP (Model Context Protocol) server, allowing Claude to use the research capabilities as tools.

### Setup for Claude Desktop

1. **Install MCP Dependencies**
   ```bash
   pip install mcp
   ```

2. **Configure Claude Desktop**
   Add the following to your Claude Desktop configuration file:

   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "research-agent": {
         "command": "python",
         "args": [
           "/path/to/ResearchLearner/mcp_server.py"
         ],
         "env": {
           "OPENAI_API_KEY": "your-openai-api-key",
           "OPENAI_MODEL": "gpt-4o",
           "OPENAI_TEMPERATURE": "0.1"
         }
       }
     }
   }
   ```

   **Important**: Update `/path/to/ResearchLearner/` with the actual path to your project directory.

3. **Set Environment Variables**
   Make sure your `.env` file contains:
   ```env
   OPENAI_API_KEY="your-openai-api-key"
   OPENAI_MODEL="gpt-4o"
   OPENAI_TEMPERATURE=0.1
   ```

4. **Restart Claude Desktop**
   After saving the configuration, restart Claude Desktop to load the MCP server.

### Available Tools in Claude

Once connected, Claude will have access to these research tools:

#### `research_topic`
Research a specific topic using arXiv papers and knowledge graph
- **Input**: `topic` (string), `max_papers` (integer, optional)
- **Example**: "Research papers about transformer architectures"

#### `query_knowledge`
Query the existing knowledge graph for information
- **Input**: `query` (string), `limit` (integer, optional)
- **Example**: "What do I know about neural networks?"

#### `analyze_paper`
Analyze a specific arXiv paper by ID
- **Input**: `paper_id` (string)
- **Example**: "Analyze paper 2301.12345"

#### `get_knowledge_summary`
Get a comprehensive knowledge summary for a topic
- **Input**: `topic` (string)
- **Example**: "Get knowledge summary for machine learning"

#### `add_research_insight`
Add a research insight to the knowledge graph
- **Input**: `insight` (string), `topic` (string), `context` (object, optional)
- **Example**: "Store this insight about attention mechanisms"

### Usage Examples with Claude

Once the MCP server is connected, you can ask Claude:

- *"Use the research agent to find papers about quantum machine learning"*
- *"Query your knowledge base for what you know about transformers"*
- *"Analyze the latest JEPA paper and add insights to your knowledge"*
- *"Get a summary of all the research you've done on reinforcement learning"*

### Troubleshooting MCP Connection

**Configuration Issues**
- Verify the path to `mcp_server.py` is correct and absolute
- Ensure Python can find all dependencies (run from activated virtual environment)
- Check that environment variables are properly set

**Connection Problems**
- Restart Claude Desktop after configuration changes
- Check Claude Desktop logs for MCP server errors
- Verify the MCP server runs independently: `python mcp_server.py`

**Permission Issues**
- Ensure the knowledge database directory `~/.research_learner/knowledge_db` is writable
- Verify OpenAI API key has sufficient permissions and credits

### MCP Server Features

The MCP server provides:
- **Stateful Knowledge**: Maintains persistent knowledge across Claude sessions
- **Resource Access**: Claude can read knowledge base and recent papers
- **Full Integration**: Access to all research agent capabilities through simple tool calls
- **Error Handling**: Graceful error handling with informative messages

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]