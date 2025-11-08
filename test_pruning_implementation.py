"""
测试 Token Pruning 实现的正确性
运行基础测试来验证实现
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_modules import TokenPruningCache, global_pruning_cache

def test_pruning_cache():
    """测试缓存逻辑"""
    print("=" * 60)
    print("测试 1: TokenPruningCache 逻辑")
    print("=" * 60)
    
    cache = TokenPruningCache()
    cache.enabled = True
    cache.denoise_token_length = 100
    cache.total_token_length = 200
    
    tests = [
        # (step, should_prune, should_cache, cache_from)
        (0, False, True, None),    # 步骤 1: 完整计算，缓存
        (1, True, False, 0),       # 步骤 2: 使用步骤 1 缓存
        (2, False, True, None),    # 步骤 3: 完整计算，缓存
        (3, True, False, 2),       # 步骤 4: 使用步骤 3 缓存
    ]
    
    for step, expected_prune, expected_cache, expected_cache_from in tests:
        cache.current_step = step
        should_prune = cache.should_prune_current_step()
        should_cache = cache.should_cache_current_step()
        cache_from = cache.get_cache_step_idx()
        
        assert should_prune == expected_prune, f"步骤 {step}: should_prune 错误"
        assert should_cache == expected_cache, f"步骤 {step}: should_cache 错误"
        assert cache_from == expected_cache_from, f"步骤 {step}: cache_from 错误"
        
        status = "Prune" if should_prune else ("Cache" if should_cache else "Normal")
        print(f"  步骤 {step + 1}: {status:6s} - should_prune={should_prune}, should_cache={should_cache}, cache_from={cache_from}")
    
    print("✅ 缓存逻辑测试通过！\n")


def test_cache_storage():
    """测试缓存存储和读取"""
    print("=" * 60)
    print("测试 2: 缓存存储和读取")
    print("=" * 60)
    
    cache = TokenPruningCache()
    cache.enabled = True
    cache.denoise_token_length = 10
    cache.total_token_length = 20
    
    # 模拟数据
    dummy_hidden_states = torch.randn(1, 20, 128)  # [B, L_total, D]
    
    # 步骤 0: 缓存
    cache.current_step = 0
    for layer_idx in range(5):  # 模拟 5 层
        cache.cache_layer_hidden_states(layer_idx, dummy_hidden_states)
    
    print(f"  已缓存 5 层数据（步骤 0）")
    
    # 步骤 1: 读取
    cache.current_step = 1
    for layer_idx in range(5):
        cached = cache.get_cached_layer_hidden_states(layer_idx)
        assert cached is not None, f"层 {layer_idx} 缓存读取失败"
        assert cached.shape == (1, 10, 128), f"层 {layer_idx} 缓存形状错误"
    
    print(f"  已读取 5 层缓存（步骤 1）")
    print("✅ 缓存存储测试通过！\n")


def test_token_split():
    """测试 token 分离逻辑"""
    print("=" * 60)
    print("测试 3: Token 分离")
    print("=" * 60)
    
    L_denoise = 100
    L_image = 50
    L_total = 150
    D = 128
    
    hidden_states = torch.randn(2, L_total, D)  # [B, L_total, D]
    
    # 分离
    denoise_hidden = hidden_states[:, :L_denoise]
    image_hidden = hidden_states[:, L_denoise:]
    
    print(f"  原始: {hidden_states.shape}")
    print(f"  去噪部分: {denoise_hidden.shape}")
    print(f"  图像部分: {image_hidden.shape}")
    
    assert denoise_hidden.shape == (2, L_denoise, D)
    assert image_hidden.shape == (2, L_image, D)
    
    # 重新合并
    reconstructed = torch.cat([denoise_hidden, image_hidden], dim=1)
    assert reconstructed.shape == hidden_states.shape
    assert torch.allclose(reconstructed, hidden_states)
    
    print("✅ Token 分离测试通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Token Pruning 实现测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_pruning_cache()
        test_cache_storage()
        test_token_split()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n可以安全使用 Token Pruning 功能。")
        print("运行: python run_with_token_pruning.py -i input.png -p \"Your prompt\"")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

