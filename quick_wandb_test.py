#!/usr/bin/env python3
# 快速WandB监控测试
import sys
sys.path.append('R2Gen-main')

from modules.wandb_logger import WandBLogger
import time

def quick_test():
    print("🧪 快速WandB测试...")
    
    logger = WandBLogger(project_name="R2Gen-Quick-Test")
    
    config = {
        'test_type': 'quick_verification',
        'batch_size': 4,
        'learning_rate': 0.001
    }
    
    logger.init_run(config, run_name="quick_test")
    
    # 记录一些测试数据
    for i in range(5):
        logger.log_training_metrics(
            epoch=1,
            batch_idx=i,
            loss=3.0 - i * 0.1,
            learning_rate=0.001
        )
        time.sleep(0.5)
    
    logger.finish()
    print("✅ 快速测试完成！")

if __name__ == '__main__':
    quick_test()
