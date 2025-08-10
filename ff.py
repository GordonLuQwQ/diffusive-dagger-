#!/usr/bin/env python3
# test_dagger.py

try:
    from agent.dagger_agent import DAggerAgent
    print("✅ DAggerAgent yesssssssssssss")
    
    # 检查类方法
    methods = [m for m in dir(DAggerAgent) if not m.startswith('_')]
    print(f"📋 yesssssssssssssssssss: {methods}")
    
    # 检查是否有必要的方法
    required_methods = ['create', 'update', 'sample']
    for method in required_methods:
        if hasattr(DAggerAgent, method):
            print(f"✅ {method} yesc")
        else:
            print(f"❌ {method} yesssssss")
            
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()