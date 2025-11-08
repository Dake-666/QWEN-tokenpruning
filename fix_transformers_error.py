"""
修复 transformers 版本兼容性问题
错误: AttributeError: 'dict' object has no attribute 'to_dict'
解决方案: 升级 transformers 到最新版本
"""
import subprocess
import sys

def fix_transformers():
    print("=" * 60)
    print("修复 transformers 兼容性问题")
    print("=" * 60)
    
    print("\n正在升级 transformers 到最新版本...")
    print("这可能需要几分钟时间...\n")
    
    try:
        # 升级 transformers
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "transformers>=4.48.0"
        ])
        print("\n✅ transformers 升级成功！")
        
        # 验证版本
        import transformers
        print(f"当前版本: {transformers.__version__}")
        
    except Exception as e:
        print(f"\n❌ 升级失败: {e}")
        print("\n请手动运行:")
        print("pip install --upgrade transformers>=4.48.0")

if __name__ == "__main__":
    fix_transformers()

