import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
def message_to_sns(message):
    # 这里可以随意添加通知手段
    logging.info(message)

