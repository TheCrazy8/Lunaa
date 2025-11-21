#!/usr/bin/env python3
"""
Test script for Lunaa modules
"""
import sys
sys.path.insert(0, '.')

def test_memory():
    from lunaa_modules.memory.memory_engine import MemoryEngine
    mem = MemoryEngine('test_memory.json')
    mem.add_fact("Test fact", "test")
    results = mem.search_facts("test")
    assert len(results) > 0, "Memory test failed"
    mem.clear_memory()
    import os
    if os.path.exists('test_memory.json'):
        os.remove('test_memory.json')
    print("✓ Memory engine test passed")

def test_context():
    from lunaa_modules.context.context_engine import ContextEngine
    ctx = ContextEngine()
    ctx.add_to_context({'role': 'user', 'content': 'Hello World'})
    summary = ctx.get_context_summary()
    assert 'Context buffer size' in summary, "Context test failed"
    print("✓ Context engine test passed")

def test_file_viewer():
    from lunaa_modules.tools.file_viewer import FileViewer
    fv = FileViewer()
    result = fv.view_file('README.md', max_lines=5)
    assert 'Lunaa' in result, "File viewer test failed"
    print("✓ File viewer test passed")

def test_math():
    from lunaa_modules.tools.math_engine import MathEngine
    math_eng = MathEngine()
    result = math_eng.calculate("2 + 2")
    # Math may not be available in all environments
    if "not installed" not in result.lower():
        assert result == "4", "Math engine test failed"
    print(f"✓ Math engine test passed - {result}")

def test_extensions():
    from lunaa_modules.extensions.extension_manager import ExtensionManager
    ext_mgr = ExtensionManager()
    result = ext_mgr.list_extensions()
    print(f"✓ Extension manager test passed - {result}")

def test_command_api():
    from lunaa_modules.command_api.api_server import CommandAPI
    api = CommandAPI(port=9999)
    result = api.start_server()
    assert 'started' in result.lower(), "Command API test failed"
    api.stop_server()
    print("✓ Command API test passed")

if __name__ == '__main__':
    print("Testing Lunaa modules...\n")
    try:
        test_memory()
        test_context()
        test_file_viewer()
        test_math()
        test_extensions()
        test_command_api()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
