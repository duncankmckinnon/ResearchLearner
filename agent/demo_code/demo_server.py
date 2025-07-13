from flask import Flask, request, jsonify, render_template, Response
import os
import sys
import requests
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to path so we can import pharmacy_bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# FastAPI server URL from environment variable
FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')
logger.info(f"Using FastAPI URL: {FASTAPI_URL}")


@app.route('/')
def index():
    """Serve the demo page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Forward the request to the FastAPI endpoint (legacy endpoint)"""
    try:
        data = request.json
        conversation_hash = data.get('conversation_hash')
        message = data.get('message')
        request_timestamp = data.get('request_timestamp')
        
        logger.info(f"Forwarding request to FastAPI: {message}")
        
        # Forward request to FastAPI endpoint
        response = requests.post(
            f"{FASTAPI_URL}/agent",
            json={
                "conversation_hash": conversation_hash,
                "request_timestamp": request_timestamp,
                "customer_message": message
            },
            timeout=120  # Increased timeout for longer operations
        )
        
        if response.status_code == 200:
            logger.info("Received successful response from FastAPI")
            return jsonify(response.json())
        else:
            logger.error(f"FastAPI returned error status: {response.status_code}")
            return jsonify({
                "response": "I apologize, but I'm having trouble processing your request. Please try again later."
            }), 500

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with FastAPI: {str(e)}")
        return jsonify({
            "response": f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "response": f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Stream chat responses with real-time updates"""
    try:
        data = request.json
        conversation_hash = data.get('conversation_hash')
        message = data.get('message')
        request_timestamp = data.get('request_timestamp')
        
        logger.info(f"Starting streaming request to FastAPI: {message}")
        
        def generate():
            try:
                # Make streaming request to FastAPI
                response = requests.post(
                    f"{FASTAPI_URL}/agent/stream",
                    json={
                        "conversation_hash": conversation_hash,
                        "request_timestamp": request_timestamp,
                        "customer_message": message
                    },
                    stream=True,
                    timeout=300  # 5 minutes timeout for streaming
                )
                
                if response.status_code == 200:
                    # Stream the response
                    for chunk in response.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            yield f"{decoded_chunk}\n"
                else:
                    # Handle error
                    error_msg = f"data: {json.dumps({'type': 'error', 'message': 'Server error occurred'})}\n\n"
                    yield error_msg
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error in streaming request: {str(e)}")
                error_msg = f"data: {json.dumps({'type': 'error', 'message': f'Connection error: {str(e)}'})}\n\n"
                yield error_msg
            except Exception as e:
                logger.error(f"Unexpected error in streaming: {str(e)}")
                error_msg = f"data: {json.dumps({'type': 'error', 'message': f'Unexpected error: {str(e)}'})}\n\n"
                yield error_msg
        
        return Response(
            generate(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
        
    except Exception as e:
        logger.error(f"Error initializing streaming: {str(e)}")
        return jsonify({
            "response": f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if FastAPI is accessible
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if response.status_code == 200:
            return jsonify({"status": "healthy", "fastapi": "connected"})
        else:
            return jsonify({"status": "unhealthy", "fastapi": "disconnected"}), 503
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the agent cache"""
    try:
        response = requests.get(f"{FASTAPI_URL}/clear_cache", timeout=10)
        if response.status_code == 200:
            return jsonify({"message": "Cache cleared successfully"})
        else:
            return jsonify({"error": "Failed to clear cache"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Knowledge Store API Proxy Endpoints

@app.route('/api/knowledge/search', methods=['POST'])
def search_knowledge():
    """Proxy for knowledge search endpoint"""
    try:
        data = request.json
        logger.info(f"Proxying knowledge search request: {data.get('query', '')}")
        
        response = requests.post(
            f"{FASTAPI_URL}/knowledge/search",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logger.error(f"Knowledge search returned error status: {response.status_code}")
            return jsonify({"error": "Knowledge search failed"}), response.status_code
            
    except Exception as e:
        logger.error(f"Error in knowledge search proxy: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/papers/<path:topic>')
def get_related_papers(topic):
    """Proxy for related papers endpoint"""
    try:
        params = {
            'limit': request.args.get('limit', 5),
        }
        logger.info(f"Proxying papers request for topic: {topic}")
        
        response = requests.get(
            f"{FASTAPI_URL}/knowledge/papers/{topic}",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logger.error(f"Papers endpoint returned error status: {response.status_code}")
            return jsonify({"error": "Papers lookup failed"}), response.status_code
            
    except Exception as e:
        logger.error(f"Error in papers proxy: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/insights/<path:topic>')
def get_research_insights(topic):
    """Proxy for research insights endpoint"""
    try:
        params = {
            'limit': request.args.get('limit', 10),
        }
        logger.info(f"Proxying insights request for topic: {topic}")
        
        response = requests.get(
            f"{FASTAPI_URL}/knowledge/insights/{topic}",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logger.error(f"Insights endpoint returned error status: {response.status_code}")
            return jsonify({"error": "Insights lookup failed"}), response.status_code
            
    except Exception as e:
        logger.error(f"Error in insights proxy: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/summary/<path:topic>')
def get_knowledge_summary(topic):
    """Proxy for knowledge summary endpoint"""
    try:
        params = {
        }
        logger.info(f"Proxying summary request for topic: {topic}")
        
        response = requests.get(
            f"{FASTAPI_URL}/knowledge/summary/{topic}",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logger.error(f"Summary endpoint returned error status: {response.status_code}")
            return jsonify({"error": "Summary lookup failed"}), response.status_code
            
    except Exception as e:
        logger.error(f"Error in summary proxy: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/memories')
def get_all_memories():
    """Proxy for all memories endpoint"""
    try:
        params = {
            'limit': request.args.get('limit', 50),
        }
        logger.info(f"Proxying memories request")
        
        response = requests.get(
            f"{FASTAPI_URL}/knowledge/memories",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logger.error(f"Memories endpoint returned error status: {response.status_code}")
            return jsonify({"error": "Memories lookup failed"}), response.status_code
            
    except Exception as e:
        logger.error(f"Error in memories proxy: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server with streaming support...")
    app.run(debug=True, load_dotenv=True, port=8080, host='0.0.0.0') 