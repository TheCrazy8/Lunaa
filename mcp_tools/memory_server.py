"""
MCP Server for Lunaa Memory Operations
"""
import json
import sys
from lunaa_modules.memory.memory_engine import MemoryEngine

def main():
    memory = MemoryEngine()
    
    # Read MCP protocol messages from stdin
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get('method')
            params = request.get('params', {})
            
            if method == 'memory/add_fact':
                result = memory.add_fact(params.get('fact'), params.get('source', 'user'))
                response = {'result': 'success', 'data': result}
            elif method == 'memory/search':
                result = memory.search_facts(params.get('query'))
                response = {'result': 'success', 'data': result}
            elif method == 'memory/get_recent':
                result = memory.get_recent_conversations(params.get('count', 10))
                response = {'result': 'success', 'data': result}
            else:
                response = {'error': f'Unknown method: {method}'}
            
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({'error': str(e)}))
            sys.stdout.flush()

if __name__ == '__main__':
    main()
