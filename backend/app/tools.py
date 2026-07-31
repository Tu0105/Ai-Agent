"""
工具定义模块
定义AI Agent可以调用的各种工具
使用LangChain的@tool装饰器定义工具
"""

import ast
import math
from datetime import datetime
from typing import Optional, Union
from langchain_core.tools import tool
import json
import random


MAX_EXPRESSION_LENGTH = 200
MAX_EXPRESSION_NODES = 64
MAX_ABSOLUTE_VALUE = 10**100


class UnsafeExpressionError(ValueError):
    """Raised when an expression is outside the calculator's small grammar."""


def safe_calculate(expression: str) -> Union[int, float]:
    """Evaluate a deliberately small arithmetic grammar without executing Python code."""
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("Expression must not be empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("Expression syntax is invalid.") from exc

    nodes_seen = 0

    def is_supported_number(value: Union[int, float]) -> bool:
        return abs(value) <= MAX_ABSOLUTE_VALUE and (isinstance(value, int) or math.isfinite(value))

    def evaluate(node: ast.AST) -> Union[int, float]:
        nonlocal nodes_seen
        nodes_seen += 1
        if nodes_seen > MAX_EXPRESSION_NODES:
            raise UnsafeExpressionError("Expression is too complex.")

        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not is_supported_number(node.value):
                raise UnsafeExpressionError("Number is outside the supported range.")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                result = left / right
            if not is_supported_number(result):
                raise UnsafeExpressionError("Result is outside the supported range.")
            return result
        raise UnsafeExpressionError("Only +, -, *, /, parentheses, and numeric literals are supported.")

    return evaluate(tree)


# ============================================
# 知识库数据（模拟数据，实际项目中可连接数据库）
# ============================================

KNOWLEDGE_BASE = {
    "退款政策": "我们的退款政策如下：\n1. 订单完成后7天内可申请全额退款\n2. 7-30天内可申请部分退款（扣除20%手续费）\n3. 30天后不支持退款\n4. 退款将在1-3个工作日内原路返回",
    "配送时间": "配送时间说明：\n1. 标准配送：3-5个工作日\n2. 加急配送：1-2个工作日（额外收取10元）\n3. 当日达：仅限部分城市，需在中午12点前下单\n4. 偏远地区可能延长1-2天",
    "会员权益": "会员权益说明：\n1. 普通会员：享受9.9折优惠\n2. 银卡会员：享受9折优惠，免运费\n3. 金卡会员：享受8.5折优惠，免运费，优先客服\n4. 钻石会员：享受8折优惠，免运费，优先客服，专属礼品",
    "如何注册": "注册流程：\n1. 点击首页右上角「注册」按钮\n2. 输入手机号并获取验证码\n3. 设置登录密码\n4. 填写基本信息完成注册\n注册成功后即可享受新人礼包！",
    "支付方式": "支持的支付方式：\n1. 支付宝\n2. 微信支付\n3. 银行卡支付\n4. 花呗/信用支付\n5. 货到付款（部分商品支持）",
    "优惠券使用": "优惠券使用规则：\n1. 每笔订单限用一张优惠券\n2. 优惠券不可叠加使用\n3. 部分商品不可使用优惠券\n4. 优惠券需在有效期内使用",
    "积分规则": "积分规则：\n1. 每消费1元累积1积分\n2. 100积分可抵扣1元\n3. 积分不可提现\n4. 积分有效期为获得后12个月",
    "客服时间": "客服工作时间：\n1. 在线客服：9:00-22:00\n2. 电话客服：9:00-21:00\n3. 周末及节假日照常服务\n4. 紧急问题可拨打24小时热线",
}


# 模拟订单数据
ORDERS_DB = {
    "ORD2024001": {
        "order_id": "ORD2024001",
        "status": "已发货",
        "product": "无线蓝牙耳机",
        "amount": 299.00,
        "create_time": "2024-01-15 10:30:00",
        "delivery_time": "2024-01-18 14:20:00",
        "express_company": "顺丰速运",
        "tracking_number": "SF1234567890"
    },
    "ORD2024002": {
        "order_id": "ORD2024002",
        "status": "处理中",
        "product": "智能手环",
        "amount": 199.00,
        "create_time": "2024-01-18 09:15:00",
        "estimated_delivery": "2024-01-21",
        "delivery_time": None,
        "express_company": None,
        "tracking_number": None
    },
    "ORD2024003": {
        "order_id": "ORD2024003",
        "status": "已完成",
        "product": "移动电源",
        "amount": 129.00,
        "create_time": "2024-01-10 16:45:00",
        "delivery_time": "2024-01-13 11:30:00",
        "express_company": "中通快递",
        "tracking_number": "ZTO9876543210"
    },
}


# FAQ数据
FAQ_DATA = {
    "如何修改密码": "修改密码步骤：\n1. 登录后进入「个人中心」\n2. 点击「账号安全」\n3. 选择「修改密码」\n4. 输入原密码和新密码\n5. 点击确认完成修改",
    "如何更换手机号": "更换手机号步骤：\n1. 进入「个人中心」-「账号安全」\n2. 点击「更换手机号」\n3. 输入新的手机号并验证\n4. 完成更换",
    "如何联系人工客服": "联系人工客服的方式：\n1. 点击右下角在线客服图标\n2. 输入「人工」转人工服务\n3. 或拨打客服热线：400-888-8888\n4. 服务时间：9:00-21:00",
    "订单如何取消": "取消订单说明：\n1. 未发货订单可自助取消\n2. 进入「我的订单」-「待发货」\n3. 点击「取消订单」并选择原因\n4. 已发货订单需等收货后申请退款",
    "商品坏了怎么办": "商品问题处理：\n1. 签收7天内可申请退换货\n2. 进入订单详情点击「申请售后」\n3. 上传商品照片并描述问题\n4. 审核通过后免费退货或换货",
}


# ============================================
# 工具函数（内部使用）
# ============================================

def _get_weather_emoji(condition: str) -> str:
    """根据天气状况返回对应的表情符号"""
    weather_emojis = {
        "晴": "☀️",
        "阴": "☁️",
        "雨": "🌧️",
        "雪": "❄️",
        "雾": "🌫️",
        "雷": "⛈️",
        "多云": "⛅",
    }
    return weather_emojis.get(condition, "🌤️")


def _generate_weather_data(city: str) -> dict:
    """生成模拟天气数据"""
    conditions = ["晴", "多云", "阴", "小雨", "晴转多云"]
    temps = list(range(15, 30))
    
    return {
        "city": city,
        "condition": random.choice(conditions),
        "temperature": random.choice(temps),
        "humidity": random.randint(40, 80),
        "wind": f"{random.randint(5, 20)}km/h",
        "aqi": random.randint(1, 150),
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ============================================
# 工具定义（使用@tool装饰器）
# ============================================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    
    Args:
        city: 城市名称，例如 "北京"、"上海"、"广州"
    
    Returns:
        str: 天气信息，包含温度、湿度、风力等
    """
    try:
        weather = _generate_weather_data(city)
        emoji = _get_weather_emoji(weather["condition"])
        
        result = f"""📍 {city}天气预报

🌡️ 温度: {weather['temperature']}°C
{emoji} 天气: {weather['condition']}
💧 湿度: {weather['humidity']}%
🌬️ 风力: {weather['wind']}
📊 空气质量指数: {weather['aqi']}

🕐 更新时间: {weather['update_time']}"""
        
        return result
    except Exception as e:
        return f"查询天气失败，请稍后重试。错误信息：{str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    执行数学计算。
    
    Args:
        expression: 数学表达式，例如 "2+3*4"、"100/5+20"、"(10+20)*3"
    
    Returns:
        str: 计算结果
    """
    try:
        # 安全检查：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "⚠️ 表达式包含非法字符，仅支持数字和基本运算符"
        
        # 使用eval进行计算（实际项目中应使用更安全的eval替代方案）
        result = safe_calculate(expression)
        
        # 格式化结果
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 8)
        
        return f"📐 计算结果\n\n expression = {expression}\n\n ✅ 结果: {result}"
    except ZeroDivisionError:
        return "⚠️ 错误：除数不能为零"
    except Exception as e:
        return f"⚠️ 计算错误，请检查表达式是否正确。错误信息：{str(e)}"


@tool
def get_current_time() -> str:
    """
    获取当前时间。
    
    Returns:
        str: 当前日期和时间
    """
    now = datetime.now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_names[now.weekday()]
    
    return f"""🕐 当前时间

📅 日期: {now.strftime('%Y年%m月%d日')}
⏰ 时间: {now.strftime('%H:%M:%S')}
📆 {weekday}

祝您有美好的一天！"""


@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库中的相关信息。
    
    Args:
        query: 搜索关键词，例如 "退款政策"、"配送时间"、"会员权益"
    
    Returns:
        str: 搜索到的相关信息
    """
    query_lower = query.lower()
    results = []
    
    # 模糊匹配
    for key, value in KNOWLEDGE_BASE.items():
        if query_lower in key.lower() or query_lower in value.lower():
            results.append(f"📖 {key}\n{value}")
    
    if results:
        return "\n\n---\n\n".join(results[:3])  # 最多返回3条结果
    else:
        return f"抱歉，暂未找到与「{query}」相关的知识库内容。\n您可以尝试以下关键词：\n• 退款政策\n• 配送时间\n• 会员权益\n• 支付方式\n• 优惠券使用"


@tool
def get_order_status(order_id: str) -> str:
    """
    查询订单状态。
    
    Args:
        order_id: 订单号，例如 "ORD2024001"
    
    Returns:
        str: 订单详细信息
    """
    # 标准化订单号
    order_id = order_id.strip().upper()
    
    if order_id not in ORDERS_DB:
        return f"""🔍 未找到订单 {order_id}

可能的原因：
• 订单号输入错误
• 订单尚未生成
• 订单已超过保留期限

💡 提示：请检查订单号是否正确，格式如：ORD2024001"""
    
    order = ORDERS_DB[order_id]
    
    status_emoji = {
        "已发货": "📦",
        "处理中": "⏳",
        "已完成": "✅",
        "已取消": "❌"
    }
    
    emoji = status_emoji.get(order["status"], "📋")
    
    result = f"""{emoji} 订单信息 - {order['status']}

🆔 订单号: {order['order_id']}
🛍️ 商品: {order['product']}
💰 金额: ¥{order['amount']:.2f}
📅 下单时间: {order['create_time']}"""
    
    if order["status"] == "已发货":
        result += f"""
🚚 快递公司: {order['express_company']}
📮 运单号: {order['tracking_number']}
📍 送达时间: {order['delivery_time']}"""
    elif order["status"] == "处理中":
        result += f"""
⏰ 预计发货: {order['estimated_delivery']}"""
    
    return result


@tool
def FAQ_answer(question: str) -> str:
    """
    回答常见问题。
    
    Args:
        question: 问题内容，例如 "如何修改密码"、"如何联系人工客服"
    
    Returns:
        str: 问题的解答
    """
    question_lower = question.lower()
    results = []
    
    # 精确匹配
    for key, value in FAQ_DATA.items():
        if question_lower in key.lower():
            results.append(f"❓ {key}\n\n{value}")
    
    # 如果精确匹配没找到，尝试模糊匹配
    if not results:
        for key, value in FAQ_DATA.items():
            # 检查关键词是否在问题中
            keywords = key.replace("如何", "").replace("？", "")
            if any(kw in question_lower for kw in keywords.split("、")):
                results.append(f"❓ {key}\n\n{value}")
    
    if results:
        return "\n\n---\n\n".join(results[:2])
    else:
        # 提供常见问题建议
        suggestions = "\n".join([f"• {q}" for q in FAQ_DATA.keys()])
        return f"""🤔 抱歉，暂未找到「{question}」的明确解答。

您可能想了解：
{suggestions}

或者输入「人工」转接人工客服为您解答。"""


@tool
def get_user_info(user_id: str) -> str:
    """
    获取用户基本信息。
    
    Args:
        user_id: 用户ID
    
    Returns:
        str: 用户信息
    """
    # 模拟用户数据
    users = {
        "U001": {"name": "张三", "phone": "138****8888", "member_level": "金卡会员", "points": 5680},
        "U002": {"name": "李四", "phone": "139****6666", "member_level": "银卡会员", "points": 2350},
        "U003": {"name": "王五", "phone": "137****5555", "member_level": "普通会员", "points": 520},
    }
    
    user_id = user_id.strip().upper()
    
    if user_id not in users:
        return f"未找到用户 {user_id} 的信息"
    
    user = users[user_id]
    
    return f"""👤 用户信息

🆔 用户ID: {user_id}
📛 姓名: {user['name']}
📱 手机: {user['phone']}
⭐ 会员等级: {user['member_level']}
🎫 积分: {user['points']}"""


@tool
def create_ticket(title: str, description: str, priority: str = "normal") -> str:
    """
    创建客服工单。
    
    Args:
        title: 工单标题
        description: 工单描述
        priority: 优先级，可选 "low", "normal", "high", "urgent"
    
    Returns:
        str: 工单创建结果
    """
    ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    priority_labels = {
        "low": "🟢 低",
        "normal": "🟡 中",
        "high": "🟠 高",
        "urgent": "🔴 紧急"
    }
    
    return f"""✅ 工单创建成功

🆔 工单号: {ticket_id}
📌 标题: {title}
📝 描述: {description}
⚡ 优先级: {priority_labels.get(priority, '🟡 中')}
🕐 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📌 提示：
• 您可以随时输入「查询工单 {ticket_id}」查看进度
• 我们的工作人员将在24小时内处理
• 如有紧急问题，请拨打客服热线"""


# ============================================
# 工具列表（用于Agent初始化）
# ============================================

TOOLS = [
    get_weather,
    calculate,
    get_current_time,
    search_knowledge,
    get_order_status,
    FAQ_answer,
    get_user_info,
    create_ticket,
]


def get_all_tools() -> list:
    """
    获取所有可用的工具
    
    Returns:
        list: 工具列表
    """
    return TOOLS


def get_tool_names() -> list:
    """
    获取所有工具的名称
    
    Returns:
        list: 工具名称列表
    """
    return [tool.name for tool in TOOLS]
